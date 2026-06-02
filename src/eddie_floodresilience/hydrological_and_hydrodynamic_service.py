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

from datetime import datetime
import json
from urllib.parse import urlencode

from pywps import BoundingBoxInput, ComplexInput, ComplexOutput, Format, LiteralInput, Process, WPSRequest
from pywps.response.execute import ExecuteResponse
from shapely import box

from src.eddie_floodresilience import tasks
from src.eddie_floodresilience.config import EnvVariable as EnvVar
from src.eddie_floodresilience.solutions.total_solutions import GLOBCOVER_CLASSES


class Whirinaki1999ScenarioProcessService(Process):
    """Class representing a WebProcessingService process for creating a flooding scenario"""

    # pylint: disable=too-few-public-methods

    def __init__(self) -> None:
        """Define inputs and outputs of the WPS process, and assign process handler."""
        # Create bounding box WPS inputs
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

        # Initialise the process
        super().__init__(
            self._handler,
            identifier="whirinaki1999",
            title="Whirinaki 1999",
            inputs=inputs,
            outputs=outputs,
        )

    @staticmethod
    def _handler(request: WPSRequest, response: ExecuteResponse) -> None:
        """
        Process handler for modelling a flood scenario

        Parameters
        ----------
        request : WPSRequest
            The WPS request, containing input parameters.
        response : ExecuteResponse
            The WPS response, containing output data.
        """
        location_geojson_str = request.inputs["location"][0].data
        landcover_type_name = request.inputs["landcover"][0].data
        print("Starting task")
        modelling_task = tasks.create_hydrological_and_hydrodynamic_model_whirinaki_1999.delay(location_geojson_str,
                                                                                               landcover_type_name)
        scenario_id = modelling_task.get()

        # Add Geoserver JSON Catalog entries to WPS response for use by Terria
        response.outputs['floodDepth'].data = json.dumps(flood_depth_catalog(scenario_id))
        response.outputs['floodedBuildings'].data = json.dumps(building_flood_status_catalog(scenario_id))


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
    dataset_name = "Building Flood Status"
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
        "name": "Flood Depth",
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
