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

"""Defines WebProcessingService processes for creating flood scenarios with hydraulic and hydrodynamic modelling."""

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
    """Abstract base class for a Process for a scenario. Children of this provide the specific task to run."""

    def __init__(self, title: str, identifier: str, task: Callable, isBaseline: bool = False) -> None:
        """Define inputs and outputs of the WPS process, and assign process handler."""
        # Create bounding box WPS inputs
        if isBaseline:
            # A very simple placeholder configuration for baseline, ideally this would be removed.
            # Inputs are required for TerriaJS though, so the front-end code would have to be fixed to allow this.
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
            ComplexOutput("landcover", "Landcover",
                          supported_formats=[Format("application/vnd.terriajs.catalog-member+json")]),
            ComplexOutput("catchmentBoundary", "CatchmentBoundary",
                          supported_formats=[Format("application/vnd.terriajs.catalog-member+json")]),
            ComplexOutput("floodDepth", "Maximum Flood Depth",
                          supported_formats=[Format("application/vnd.terriajs.catalog-member+json")]),
            ComplexOutput("injectionPoints", "River Flows",
                          supported_formats=[Format("application/vnd.terriajs.catalog-member+json")]),
            ComplexOutput("floodedBuildings", "Flooded Buildings",
                          supported_formats=[Format("application/vnd.terriajs.catalog-member+json")])
        ]
        # Add outputs that only make sense for non-baseline scenarios.
        if not isBaseline:
            outputs.append(
                ComplexOutput("depthDifference", "Difference in Flood Depth to Baseline",
                              supported_formats=[Format("application/vnd.terriajs.catalog-member+json")])
            )

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

    def _handler(request: WPSRequest, response: ExecuteResponse) -> None:
        """
        Process handler for modelling a flood scenario

        Parameters
        ----------
        request : WPSRequest
            The WPS request, containing input parameters.
        response : ExecuteResponse
            The WPS response, containing output data.

        Returns
        -------
        Callable
            The WPS handler function.
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

        scenario_name = "Baseline" if is_baseline else str(scenario_id)

        # Add Geoserver JSON Catalog entries to WPS response for use by Terria
        response.outputs['landcover'].data = json.dumps(landcover_catalog(scenario_id, scenario_name))
        response.outputs['catchmentBoundary'].data = json.dumps(catchment_boundary_catalog(scenario_id, scenario_name))
        response.outputs['floodDepth'].data = json.dumps(flood_depth_catalog(scenario_id, scenario_name))
        response.outputs['injectionPoints'].data = json.dumps(
            hydrograph_injection_point_catalog(scenario_id, scenario_name)
        )
        response.outputs['floodedBuildings'].data = json.dumps(
            building_flood_status_catalog(scenario_id, scenario_name)
        )
        if not is_baseline:
            response.outputs['depthDifference'].data = json.dumps(depth_difference_catalog(scenario_id, scenario_name))

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


def building_flood_status_catalog(scenario_id: int, scenario_name: str) -> dict:
    """
    Create a dictionary in the format of a terria js catalog json for the building flood status layer.

    Parameters
    ----------
    scenario_id : int
        The ID of the scenario to create the catalog item for.
    scenario_name : str
        The name of the scenario to create the catalog item for.


    Returns
    ----------
    dict
        The TerriaJS catalog item JSON for the building flood status layer.
    """
    dataset_name = f"Building Flood Status - {scenario_name}"
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


def flood_depth_catalog(scenario_id: int, scenario_name: str) -> dict:
    """
    Create a dictionary in the format of a terria js catalog json for the flood depth layer.

    Parameters
    ----------
    scenario_id : int
        The ID of the scenario to create the catalog item for.
    scenario_name : str
        The name of the scenario to create the catalog item for.

    Returns
    ----------
    dict
        The TerriaJS catalog item JSON for the flood depth layer.
    """
    gs_flood_model_workspace = f"{EnvVar.POSTGRES_DB}-dt-model-outputs"
    layer_name = f"output_{scenario_id}"
    style_name = "plasma_0_3m"
    display_name = f"Flood Depth - {scenario_name}"
    return _wms_depth_catalog(
        scenario_id,
        workspace_name=gs_flood_model_workspace,
        layer_name=layer_name,
        display_name=display_name,
        style_name=style_name
    )


def depth_difference_catalog(scenario_id: int, scenario_name: str) -> dict:
    """
    Create a dict in the format of a terria js catalog json for the difference between scenario and baseline depth.

    Parameters
    ----------
    scenario_id : int
        The ID of the scenario to create the catalog item for.
    scenario_name : str
        The name of the scenario to create the catalog item for.

    Returns
    ----------
    dict
        The TerriaJS catalog item JSON for the difference layer.
    """
    return _wms_depth_catalog(
        scenario_id,
        workspace_name=f"{EnvVar.POSTGRES_DB}-dt-model-outputs",
        layer_name=f"diff_{scenario_id}",
        display_name=f"Difference from baseline to {scenario_name}",
        style_name="difference_r_b_0_2"
    )


def _wms_depth_catalog(
    scenario_id: int, workspace_name: str, layer_name: str, display_name: str, style_name: str
) -> dict:
    """
    Build a WMS catalog item JSON based on the given parameters, for a raster about water depth.

    Parameters
    ----------
    scenario_id : int
        The ID of the scenario to create the catalog item for.
    workspace_name : str
        The name of the GeoServer workspace the layer lives in.
    layer_name : str
        The name of the WMS layer within GeoServer.
    display_name : str
        The label to add to the catalog item in the front-end.
    style_name : str
        The name of the GeoServer style to display the layer with.

    Returns
    -------
    dict
        The TerriaJS catalog item JSON for the WMS Rster layer.
    """
    gs_service_url = f"{EnvVar.GEOSERVER_HOST}:{EnvVar.GEOSERVER_PORT}/geoserver/{workspace_name}/ows"
    fully_qualified_layer = f"{workspace_name}:{layer_name}"
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
        "layer": fully_qualified_layer,
        "style": style_name,
        "transparent": "true",
        "LEGEND_OPTIONS": "hideEmptyRules:true;"
                          "forceLabels:on;"
                          "labelMargin:5;"
                          "fontColor:0xffffff;"
                          "fontStyle:bold;"
                          "fontAntiAliasing:true;"
    }
    legend_url = f"{gs_service_url}?{urlencode(legend_url_params)}"

    return {
        "type": "wms",
        "name": display_name,
        "url": gs_service_url,
        "layers": fully_qualified_layer,
        "styles": style_name,
        "featureInfoTemplate": {
            "name": display_name,
            "template": flood_depth_infobox_template.format(flood_scenario_id=scenario_id, layer_name=layer_name),
        },
        "legends": [{
            "title": display_name,
            "url": legend_url,
            "urlMimeType": "image/png"
        }],
    }


def catchment_boundary_catalog(scenario_id: int, scenario_name: str) -> dict:
    """
    Create a dictionary in the format of a terria js catalog json for the building flood status layer.

    Parameters
    ----------
    scenario_id : int
        The ID of the scenario to create the catalog item for.
    scenario_name : str
        The name of the scenario to create the catalog item for.

    Returns
    ----------
    dict
        The TerriaJS catalog item JSON for the building flood status layer.
    """
    dataset_name = f"Catchment Boundary - {scenario_name}"
    gs_building_workspace = f"{EnvVar.POSTGRES_DB}-intermediate-wflow"
    gs_building_url = f"{EnvVar.GEOSERVER_HOST}:{EnvVar.GEOSERVER_PORT}/geoserver/{gs_building_workspace}/ows"

    return {
        "type": "wfs",
        "name": dataset_name,
        "url": gs_building_url,
        "typeNames": f"{gs_building_workspace}:wflow_catchment_boundary",
        "parameters": {
            "CQL_FILTER": f"flood_model_id = {scenario_id}",
        },
        "styles": [
            {
                "id": "Catchment",
                "color": {
                    "nullColor": "rgba(0,0,0,0)"
                },
                "outline": {
                    "null": {
                        "color": "rgba(201,0,0,1)", "width": 2
                    }
                }
            }
        ],
        "activeStyle": "Catchment",
    }


def landcover_catalog(scenario_id: int, scenario_name: str) -> dict:
    """
    Create a dictionary in the format of a terria js catalog json for the landcover layer.

    Parameters
    ----------
    scenario_id : int
        The ID of the scenario to create the catalog item for.
    scenario_name : str
        The name of the scenario to create the catalog item for.

    Returns
    ----------
    dict
        The TerriaJS catalog item JSON for the building flood status layer.
    """
    gs_intermediate_workspace = f"{EnvVar.POSTGRES_DB}-intermediate-wflow"
    gs_landcover_url = f"{EnvVar.GEOSERVER_HOST}:{EnvVar.GEOSERVER_PORT}/geoserver/{gs_intermediate_workspace}/ows"
    layer_name = f"{gs_intermediate_workspace}:landcover_{scenario_id}"

    return {
        "type": "wms",
        "name": f"Landcover - {scenario_name}",
        "url": gs_landcover_url,
        "layers": layer_name,
        "styles": "landcover",
    }


def hydrograph_injection_point_catalog(scenario_id: int, scenario_name: str) -> dict:
    """
    Create a dictionary in the format of a terria js catalog json for the injection points, with hydrographs.

    Parameters
    ----------
    scenario_id : int
        The ID of the scenario to create the catalog item for.
    scenario_name : str
        The name of the scenario to create the catalog item for.

    Returns
    ----------
    dict
        The TerriaJS catalog item JSON for the building flood status layer.
    """
    gs_flood_inputs_workspace = f"{EnvVar.POSTGRES_DB}-flood-model-inputs"
    gs_flood_inputs_url = f"{EnvVar.GEOSERVER_HOST}:{EnvVar.GEOSERVER_PORT}/geoserver/{gs_flood_inputs_workspace}/ows"

    # Open and read HTML template for plot in infobox for TerriaJS
    with open("./src/eddie_floodresilience/flood_model/templates/plot_infobox_template.html",
              encoding="utf-8") as file:
        plot_infobox_template = file.read()

    # Fill in plot template vars. Some parts are double escaped so they can be used as moustache templates in TerriaJS.
    plot_title = f"Hydrograph — Injection Point {{{{FID}}}} ({scenario_name})"
    csv_src = f"{EnvVar.BACKEND_URL}/hydrographs/scenarios/{scenario_id}/features/{{{{FID}}}}"
    plot_infobox_template = plot_infobox_template.format(
        plot_title=plot_title,
        csv_src=csv_src,
        x_axis="Time",
        y_axis="Flow",
        y_units="Flow (m3/s)"
    )

    return {
        "type": "wfs",
        "name": f"Hydrographs - {scenario_name}",
        "url": gs_flood_inputs_url,
        "typeNames": f"{gs_flood_inputs_workspace}:injection_points",
        "parameters": {
            "CQL_FILTER": f"model_output_id={scenario_id}",
        },
        "featureInfoTemplate": {
            # Double-escaped so that after python string formatting it is still mustache formatted.
            "name": f"Hydrograph - {scenario_name} - {{{{FID}}}}",
            "template": plot_infobox_template
        }
    }
