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
from src.eddie_floodresilience.flood_model import bg_flood_model, process_hydro_dem
from src.eddie_floodresilience.run_all import DEFAULT_MODULES_TO_PARAMETERS
from src.eddie_floodresilience.config import EnvVariable
from src.eddie_floodresilience import hydrological_and_hydrodynamic_pipeline

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


@app.task(base=OnFailureStateTask)
def create_hydrological_and_hydrodynamic_model_whirinaki_1999(location_geojson: str, landcover_name: str) -> int:
    landcover_scenario_gdf = gpd.read_file(location_geojson, driver="GeoJSON")
    landcover_scenario_gdf["landcover_name"] = landcover_name
    hydrological_and_hydrodynamic_pipeline.main(landcover_scenario_gdf)
    return -1

