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
from typing import List, NamedTuple

import geopandas as gpd
from celery.signals import setup_logging as celery_setup_logging_signal

from eddie.digitaltwin import cache_new_results, check_cache_results
from eddie.digitaltwin.utils import setup_logging, LogLevel
from eddie.tasks import OnFailureStateTask, app  # pylint: disable=cyclic-import
from src.eddie_floodresilience import hydrological_and_hydrodynamic_pipeline
from src.eddie_floodresilience.hydrological.wflow_serve_data_generator import get_hydrograph_csv


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


@celery_setup_logging_signal.connect
def configure_celery_logging(*_args: tuple, **_kwargs: dict) -> None:  # pylint: disable=missing-param-doc
    """Set up celery logging to use the same logging as if running the modules directly."""
    setup_logging(LogLevel.INFO)


@app.task(base=OnFailureStateTask)
def cache_results(flood_model_id: int, scenario_options: dict) -> int:
    """
    Task to cache the scenario options used to generate an existing model with the given model id, for faster retrieval.

    Parameters
    ----------
    flood_model_id : int
        The database id of the existing model output to attach the cached parameters to.
    scenario_options : dict
        The input parameters to the model to cache, which must match for later retrieval.

    Returns
    -------
    int
        model_id re-returned to allow method chaining.
    """
    cache_new_results.main(flood_model_id, scenario_options)
    return flood_model_id


@app.task(base=OnFailureStateTask)
def check_cache(scenario_options: dict) -> int | None:
    """
    Check cache table for model output generated from identical scenario_options, and finds matching model ID.

    Parameters
    ----------
    scenario_options : dict
        The model input parameters, which must match exactly with the cached results for a positive match.

    Returns
    -------
    int | None
        Returns the matching model_id if a match is found. Otherwise, None.
    """
    flood_model_id = check_cache_results.main(scenario_options)
    return flood_model_id


@app.task(base=OnFailureStateTask)
def create_hydrological_and_hydrodynamic_model_whirinaki_1999(
    location_geojson: str | None,
    landcover_name: str | None
) -> int:
    """
    Task to run a hydrological and hydronynamic model for Whirinaki.

    Parameters
    ----------
    location_geojson: str | None
        A GeoJSON string with polygons dictating where to change landcover. # TODO combine these params into one
    landcover_name: str | None
        The landcover type to change landcover to.

    Returns
    -------
    int
        The resultant flood model output ID.
    """
    landcover_scenario_gdf = read_location_geojson(location_geojson, landcover_name)
    flood_model_output_id = hydrological_and_hydrodynamic_pipeline.whirinaki(landcover_scenario_gdf)
    return flood_model_output_id


@app.task(base=OnFailureStateTask)
def create_hydrological_and_hydrodynamic_model_mataura_2020(location_geojson: str, landcover_name: str) -> int:
    """
    Task to run a hydrological and hydronynamic model for Mataura.

    Parameters
    ----------
    location_geojson: str | None
        A GeoJSON string with polygons dictating where to change landcover. # TODO combine these params into one
    landcover_name: str | None
        The landcover type to change landcover to.

    Returns
    -------
    int
        The resultant flood model output ID.
    """
    landcover_scenario_gdf = read_location_geojson(location_geojson, landcover_name)
    flood_model_output_id = hydrological_and_hydrodynamic_pipeline.mataura(landcover_scenario_gdf)
    return flood_model_output_id


@app.task(base=OnFailureStateTask)
def read_hydrograph_data(flood_model_id: int, injection_point_feature_id: str) -> str:
    """
    Task to run read hydrograph data for an injection point.

    Parameters
    ----------
    flood_model_id: str
        The flood model output ID to find query hydrograph data for.
    injection_point_feature_id: str
        The FID of the specific injection point to query hydrograph data for.

    Returns
    -------
    str
        CSV formatted hydrograph data for an injection point.
    """
    return get_hydrograph_csv(flood_model_id, injection_point_feature_id)


def read_location_geojson(location_geojson: str | None, landcover_name: str | None) -> gpd.GeoDataFrame | None:
    """
    Read a GeoJSON string and a landcover name into a GeoDataFrame,

    Parameters
    ----------
    location_geojson: str | None
        A GeoJSON string with polygons dictating where to change landcover. # TODO combine these params into one
    landcover_name: str | None
        The landcover type to change landcover to.

    Returns
    -------
    gpd.GeoDataFrame
        The GeoDataFrame containing polygons with a column named "landcover_name".

    """
    if location_geojson is None or landcover_name is None:
        return None
    landcover_scenario_gdf = gpd.read_file(location_geojson, driver="GeoJSON")
    landcover_scenario_gdf["landcover_name"] = landcover_name
    return landcover_scenario_gdf
