# # -*- coding: utf-8 -*-
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

"""Defines PyWPS WebProcessingService process for creating a flooding scenario with hydraluic and hydrodynamic modelling."""

import json
from abc import ABC
from typing import Callable
from urllib.parse import urlencode

from celery import Task
from pywps import ComplexInput, ComplexOutput, Format, LiteralInput, Process, WPSRequest
from pywps.response.execute import ExecuteResponse

from src.eddie_floodresilience import tasks
from src.eddie_floodresilience.config import EnvVariable as EnvVar
from src.eddie_floodresilience.solutions.total_solutions import GLOBCOVER_CLASSES


class PredefinedScenario(Process, ABC):
    def __init__(self, title: str, identifier: str, task: Callable, isBaseline=False) -> None:
        """Define inputs and outputs of the WPS process, and assign process handler."""

        # Create bounding box WPS inputs
        if isBaseline:
            # A very simple placeholder configuration for baseline
            inputs = [LiteralInput(
                "options",
                "Options",
                data_type="integer",
                allowed_values=[0, 1]
            ), ]
        else:
            inputs = [
                ComplexInput(
                    'location',
                    'New Land Cover Area',
                    supported_formats=[
                        Format(mime_type='application/vnd.geo+json',
                               schema='http://geojson.org/geojson-spec.html#geojson')],
                    workdir='workdir'
                ),
                LiteralInput(
                    "landcover",
                    "Landcover Class",
                    data_type="string",
                    allowed_values=list(GLOBCOVER_CLASSES.keys())
                ),
            ]
        # Create area WPS outputs
        outputs = [
            ComplexOutput("floodDepth", "Maximum Flood Depth",
                          supported_formats=[Format("application/vnd.terriajs.catalog-member+json")]),
            ComplexOutput("floodedBuildings", "Flooded Buildings",
                          supported_formats=[Format("application/vnd.terriajs.catalog-member+json")])
        ]

        handler = handler_for_task(task, isBaseline)
        # Initialise the process
        super().__init__(
            handler,
            identifier=identifier,
            title=title,
            inputs=inputs,
            outputs=outputs,
        )


def handler_for_task(task: Task, is_baseline: bool = False) -> Callable:
    """
    Create a process handler for a given task.

    Parameters
    ----------
    task : Task
        The callback function to be executed as a task.
    is_baseline : bool = False
        Whether the scenario is configurable or a baseline. If it is a baseline then we do not have to read the inputs.

    Returns
    -------
    Callable
        The WPS handler function.
    """

    def _handler(request: WPSRequest, response: ExecuteResponse):
        """
                Process handler for modelling a flood scenario

                Parameters
                ----------
                request : WPSRequest
                    The WPS request, containing input parameters.
                response : ExecuteResponse
                    The WPS response, containing output data.
                """
        if is_baseline:
            # Inputs can be ignored in a baseline
            location_geojson_str = None
            landcover_type_name = None
        else:
            # Read the inputs
            location_geojson_str = request.inputs["location"][0].data
            landcover_type_name = request.inputs["landcover"][0].data

        # Check if scenario is already cached
        cache_dict = {
            "task": task.name,
            "location_geojson_str": location_geojson_str,
            "landcover_type_name": landcover_type_name,
        }
        check_cache_task = tasks.check_cache.delay(cache_dict)
        scenario_id = check_cache_task.get()

        # Run the task callback if its needed
        if scenario_id is None:
            modelling_task = task.delay(location_geojson_str, landcover_type_name)
            scenario_id = modelling_task.get()
            tasks.cache_results.delay(scenario_id, cache_dict)

        # Add Geoserver JSON Catalog entries to WPS response for use by Terria
        response.outputs['floodDepth'].data = json.dumps(flood_depth_catalog(scenario_id))
        response.outputs['floodedBuildings'].data = json.dumps(building_flood_status_catalog(scenario_id))

    return _handler


class Whirinaki1999ScenarioProcessService(PredefinedScenario):
    """Class representing a WebProcessingService process for creating a flooding scenario for Whirinaki"""

    # pylint: disable=too-few-public-methods

    def __init__(self) -> None:
        """Define inputs and outputs of the WPS process, and assign process handler."""
        title = "Whirinaki 1999"
        identifier = "whirinaki1999"
        task = tasks.create_hydrological_and_hydrodynamic_model_whirinaki_1999
        super().__init__(title, identifier, task)


class Whirinaki1999BaselineProcessService(PredefinedScenario):
    """Class representing a WebProcessingService process for creating a flooding baseline for Whirinaki"""

    # pylint: disable=too-few-public-methods

    def __init__(self) -> None:
        """Define inputs and outputs of the WPS process, and assign process handler."""
        title = "Whirinaki 1999 Baseline"
        identifier = "whirinaki1999baseline"
        task = tasks.create_hydrological_and_hydrodynamic_model_whirinaki_1999
        super().__init__(title, identifier, task, isBaseline=True)


class Mataura2020ScenarioProcessService(PredefinedScenario):
    """Class representing a WebProcessingService process for creating a flooding scenario for Mataura"""

    # pylint: disable=too-few-public-methods

    def __init__(self) -> None:
        """Define inputs and outputs of the WPS process, and assign process handler."""
        title = "Mataura 2020"
        identifier = "mataura2020"
        task = tasks.create_hydrological_and_hydrodynamic_model_mataura_2020
        super().__init__(title, identifier, task)


class Mataura2020BaselineProcessService(PredefinedScenario):
    """Class representing a WebProcessingService process for creating a flooding scenario for Whirinaki"""

    # pylint: disable=too-few-public-methods

    def __init__(self) -> None:
        """Define inputs and outputs of the WPS process, and assign process handler."""
        title = "Mataura 2020 Baseline"
        identifier = "mataura2020baseline"
        task = tasks.create_hydrological_and_hydrodynamic_model_mataura_2020
        super().__init__(title, identifier, task, isBaseline=True)


def building_flood_status_catalog(scenario_id: int) -> dict:
    """
    Create a dictionary in the format of a terria js catalog json for the building flood status layer.

    Parameters
    ----------
    scenario_id : int
        The ID of the scenario to create the catalog item for.

    Returns
    ----------
    dict
        The TerriaJS catalog item JSON for the building flood status layer.
    """
    dataset_name = f"Building Flood Status - {scenario_id}"
    gs_building_workspace = f"{EnvVar.POSTGRES_DB}-buildings"
    gs_building_url = f"{EnvVar.GEOSERVER_HOST}:{EnvVar.GEOSERVER_PORT}/geoserver/{gs_building_workspace}/ows"

    flooded_color = "darkred"
    non_flooded_color = "darkgreen"
    return {
        "type": "wfs",
        "name": dataset_name,
        "url": gs_building_url,
        "typeNames": f"{gs_building_workspace}:building_flood_status",
        "parameters": {
            "viewparams": f"scenario:{scenario_id}",
        },
        "maxFeatures": 300000,
        "styles": [{
            "id": "is_flooded",
            "title": dataset_name,
            "color": {
                "mapType": "enum",
                "colorColumn": "is_flooded_int",
                "legend": {
                    "title": dataset_name,
                    "items": [
                        {
                            "title": "Non-Flooded",
                            "color": non_flooded_color
                        },
                        {
                            "title": "Flooded",
                            "color": flooded_color
                        }
                    ]
                },
                "enumColors": [
                    {
                        "value": "0",
                        "color": non_flooded_color
                    },
                    {
                        "value": "1",
                        "color": flooded_color
                    }
                ]
            },
            "outline": {
                "null": {
                    "width": 0
                }
            }
        }],
        "activeStyle": "is_flooded"
    }


def flood_depth_catalog(scenario_id: int) -> dict:
    """
    Create a dictionary in the format of a terria js catalog json for the flood depth layer.

    Parameters
    ----------
    scenario_id : int
        The ID of the scenario to create the catalog item for.

    Returns
    ----------
    dict
        The TerriaJS catalog item JSON for the flood depth layer.
    """
    gs_flood_model_workspace = f"{EnvVar.POSTGRES_DB}-dt-model-outputs"
    gs_flood_url = f"{EnvVar.GEOSERVER_HOST}:{EnvVar.GEOSERVER_PORT}/geoserver/{gs_flood_model_workspace}/ows"
    layer_name = f"{gs_flood_model_workspace}:output_{scenario_id}"
    style_name = "viridis_raster"
    # Open and read HTML/mustache template file for infobox
    with open("./src/eddie_floodresilience/flood_model/templates/flood_depth_infobox.mustache",
              encoding="utf-8") as file:
        flood_depth_infobox_template = file.read()
    # Parameters for the Geoserver GetLegendGraphic request
    legend_url_params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetLegendGraphic",
        "format": "image/png",
        "sld_version": "1.1.0",
        "layer": layer_name,
        "style": style_name,
        "transparent": "true",
        "LEGEND_OPTIONS": "hideEmptyRules:true;"
                          "forceLabels:on;"
                          "labelMargin:5;"
                          "fontColor:0xffffff;"
                          "fontStyle:bold;"
                          "fontAntiAliasing:true;"
    }
    legend_url = f"{gs_flood_url}?{urlencode(legend_url_params)}"

    return {
        "type": "wms",
        "name": f"Flood Depth - {scenario_id}",
        "url": gs_flood_url,
        "layers": layer_name,
        "styles": style_name,
        "featureInfoTemplate": {
            "name": f"Flood depth - {scenario_id}",
            "template": flood_depth_infobox_template.format(flood_scenario_id=scenario_id)
        },
        "legends": [{
            "title": "Flood Depth",
            "url": legend_url,
            "urlMimeType": "image/png"
        }],
    }
