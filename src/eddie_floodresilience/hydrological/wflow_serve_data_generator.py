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

"""Serves data for WFlow scenarios, such as landcover and catchment boundaries."""

import logging
from pathlib import Path

from osgeo import gdal
import geopandas as gpd
import rioxarray as rxr

from eddie import geoserver as gs
from eddie.digitaltwin import setup_environment
from eddie.digitaltwin.utils import setup_logging
from eddie.geoserver.raster_layers import CoverageDimension
from src.eddie_floodresilience.config import EnvVariable
from src.eddie_floodresilience.hydrological.wflow_data_catalog_generator import find_landcover_file

setup_logging()
log = logging.getLogger(__name__)


class WflowServeDataGenerator:
    r"""
    Class to serve data for a Wflow scenario, such as landcover and catchment boundaries.

    Attributes
    ----------
    hydromt_path : Path
        A directory to where all necessary files are stored to run wflow model
    wflow_model_path : Path
        A directory to where the data_catalog.yml is stored for WFlow
    landcover : str
        A string to identify the landcover. Meets the approximate regex "(globcover(_\d\d\d.tif)?)|(lcdb(_\d\d\d.tif))"
    flood_model_output_id : int
        The ID of the flood model scenario that has been created using this wflow model
    """  # pylint: disable=too-few-public-methods

    def __init__(self, hydromt_path: Path, wflow_model_path: Path, landcover: str, flood_model_output_id: int) -> None:
        r"""
        Define paths and variables relating to serving a Wflow scenario.

        Parameters
        ----------
        hydromt_path : Path
            A directory to where all necessary files are stored to run wflow model
        wflow_model_path : Path
            A directory to where the data_catalog.yml is stored for WFlow
        landcover : str
            A string to identify the landcover.
            Meets the approximate regex "(globcover(_\d\d\d.tif)?)|(lcdb(_\d\d\d.tif))"
        flood_model_output_id : int
            The ID of the flood model scenario that has been created using this wflow model
        """
        self.hydromt_path = hydromt_path
        self.wflow_model_path = wflow_model_path
        self.landcover = landcover
        self.flood_model_output_id = flood_model_output_id

    def serve_data(self) -> None:
        """
        Save the data relating to the Wflow scenario to GeoServer and serve them.

        Raises
        ------
        KeyError
            If the CRS of the catchment file is not defined.
        """
        if not EnvVariable.IS_GEOSERVER_ACTIVE:
            return

        # Set up geoserver workspace
        db_name = EnvVariable.POSTGRES_DB
        workspace_name = f"{db_name}-intermediate-wflow"

        # Read catchment polygon
        catchment_file = self.wflow_model_path / "wflow_test_full/staticgeoms/basins.geojson"
        catchment_poly = gpd.read_file(catchment_file)
        if catchment_poly.crs.to_epsg() is None:
            raise KeyError(f"CRS is not defined in EPSG# form in vector file {catchment_file}.")
        catchment_poly = catchment_poly.to_crs(epsg=2193)

        log.info(f"Adding catchment file '{catchment_file}' to GeoServer.")
        self._serve_catchment_boundary(workspace_name, catchment_poly)

        # Read and serve scenario landcover file
        self._serve_landcover_file(workspace_name, catchment_poly)

    def _serve_landcover_file(self, workspace_name: str, catchment_poly: gpd.GeoDataFrame) -> None:
        """
        Serve the landcover scenario raster.

        Parameters
        ----------
        workspace_name : str
            The name of the GeoServer workspace to serve the landcover from.
        catchment_poly : gpd.GeoDataFrame
            The catchment polygon to clip the landcover raster to before serving to reduce space.
        """
        is_baseline = self.landcover == "globcover"
        landcover_file = find_landcover_file(self.wflow_model_path, self.hydromt_path, self.landcover, is_baseline)

        tmp_dir = Path("tmp/gtiff") / self.hydromt_path.name
        tmp_dir.mkdir(parents=True, exist_ok=True)
        clipped_path = tmp_dir / landcover_file.name

        with rxr.open_rasterio(landcover_file) as src:
            catchment = catchment_poly.to_crs(src.rio.crs).geometry
            clipped_landcover = src.rio.clip(catchment, drop=True)
            clipped_landcover.rio.to_raster(clipped_path)

        layer_name = f"landcover_{self.flood_model_output_id}"
        if layer_name not in gs.raster_layers.get_workspace_raster_layers(workspace_name):
            coverage_dimensions = [CoverageDimension(layer_name, "landcover_class", "Int32")]
            gs.add_gtiff_to_geoserver(clipped_path, workspace_name, layer_name, coverage_dimensions)
        # Delete tmp clipped file
        clipped_path.unlink()

    def _serve_catchment_boundary(self, workspace_name: str, catchment_poly: gpd.GeoDataFrame) -> None:
        """
        Serve the catchment boundary polygon.

        Parameters
        ----------
        workspace_name : str
            The name of the GeoServer workspace to serve the landcover from.
        catchment_poly : gpd.GeoDataFrame
            The catchment polygon to serve.
        """
        # Create the database geoserver store
        data_store = gs.create_main_db_store(workspace_name)
        # Extract the geometry of the catchment area
        catchment_poly["flood_model_id"] = self.flood_model_output_id
        catchment_poly = catchment_poly[["flood_model_id", "geometry"]]

        engine = setup_environment.get_database()
        with engine.connect() as conn:
            # Create the 'wflow_catchment_boundary' table if it doesn't exist
            table_name = "wflow_catchment_boundary"
            catchment_poly.to_postgis(table_name, conn, if_exists="append", index=False)
            gs.create_datastore_layer(conn, workspace_name, data_store, table_name)
