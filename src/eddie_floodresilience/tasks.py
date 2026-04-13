# -*- coding: utf-8 -*-
# Copyright © 2021-2026 Geospatial Research Institute Toi Hangarau
# LICENSE: https://github.com/GeospatialResearch/Digital-Twins/blob/master/LICENSE
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Runs backend tasks using Celery. Allowing for multiple long-running tasks to complete in the background.
Allows the frontend to send tasks and retrieve status later.
"""
import logging
from typing import Dict, List, NamedTuple, Union

from celery import result, signals
from celery.worker.consumer import Consumer
import geopandas as gpd
from pyproj import Transformer
import xarray

from eddie.digitaltwin import cache_new_results, check_cache_results, retrieve_from_instructions, setup_environment
from eddie.digitaltwin.utils import setup_logging
from eddie.tasks import OnFailureStateTask, add_base_data_to_db, app, wkt_to_gdf  # pylint: disable=cyclic-import
from src.eddie_floodresilience.dynamic_boundary_conditions.rainfall import main_rainfall
from src.eddie_floodresilience.dynamic_boundary_conditions.river import main_river
from src.eddie_floodresilience.dynamic_boundary_conditions.tide import main_tide_slr
from src.eddie_floodresilience.example_005 import hydro_combination_path, width_rate_control
from src.eddie_floodresilience.flood_model import bg_flood_model, process_hydro_dem
from src.eddie_floodresilience.run_all import DEFAULT_MODULES_TO_PARAMETERS
from src.eddie_floodresilience.config import EnvVariable

setup_logging()
log = logging.getLogger(__name__)


class DepthTimePlot(NamedTuple):
    """
    Represents the depths over time for a particular pixel location in a raster.
    Uses tuples and lists instead of Arrays or Dataframes because it needs to be easily serializable when communicating
    over message_broker.

    Attributes
    ----------
    depths : List[float]
        A list of all of the depths in m for the pixel. Parallels the times list
    times : List[float]
        A list of all of the times in s for the pixel. Parallels the depts list
    """

    depths: List[float]
    times: List[float]


@signals.worker_ready.connect
def on_startup(sender: Consumer, **_kwargs: None) -> None:  # pylint: disable=missing-param-doc
    """
    Initialise database, runs when Celery instance is ready.

    Parameters
    ----------
    sender : Consumer
        The Celery worker node instance
    """
    with sender.app.connection() as conn:
        # Gather area of interest from file.
        aoi_wkt = gpd.read_file("selected_polygon.geojson").to_crs(4326).geometry[0].wkt
        # Send a task to initialise this area of interest.
        base_data_parameters = DEFAULT_MODULES_TO_PARAMETERS[retrieve_from_instructions]
        sender.app.send_task("eddie.tasks.add_base_data_to_db", args=[aoi_wkt, base_data_parameters], connection=conn)
        # Send a task to ensure lidar datasets are evaluated.
        sender.app.send_task("src.eddie_floodresilience.tasks.ensure_lidar_datasets_initialised")


def create_model_whirinaki_1999() -> result.GroupResult:
    """
    Create a model for the area using series of chained (sequential) sub-tasks.

    Returns
    -------
    result.GroupResult
        The task result for the long-running group of tasks. The task ID represents the final task in the group.
    """
    hydro_combination_path = EnvVariable.HYDRO_COMBINATION_PATH
    outlet_gauge_locations_filename = EnvVariable.OUTLET_GAUGE_LOCATIONS_FILENAME

    return (
        preprocess_terrain_data.si(hydro_combination_path, outlet_gauge_locations_filename, width_rate_control=1/20)
    )()


@app.task(base=OnFailureStateTask)
def preprocess_terrain_data(
    terrain_path: Path,
    outlet_gauge_locations_filename: str,
    resolution: float = 0.00045,
    threshold: int = 1000,
    width_rate_control: float = 2,
    discharge_rate_control: float = 1
) -> None:
    """
    Task to ensure hydrologically-conditioned DEM is processed for the given area and added to the database.

    Parameters
    ----------
    selected_polygon_wkt : str
        The polygon defining the selected area to process the DEM for. Defined in WKT form.
    """
    generator = TerrainDataWflowGenerator(
        terrain_path,
        outlet_gauge_locations_filename,
        resolution,
        threshold,
        width_rate_control,
        discharge_rate_control
    )
    generator.terrain_for_wflow_generator()
