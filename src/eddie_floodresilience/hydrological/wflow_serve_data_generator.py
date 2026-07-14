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
from ast import literal_eval
from io import StringIO
from pathlib import Path

import geopandas as gpd
import pandas as pd
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
    polygons: gpd.GeoDataFrame | None
        Polygons that are used to change the landcover information.
        This polygon dataframe has 'landcover' column with new values
    landcover_mapping_type : str
        Name of landcover dataset - globcover or lcdb
    scenario_and_id_folder : Path
        Directory to the scenario folder name with ID
    flood_model_output_id : int
        The ID of the flood model scenario that has been created using this wflow model
    """  # pylint: disable=too-few-public-methods

    def __init__(
        self,
        hydromt_path: Path,
        polygons: gpd.GeoDataFrame | None,
        landcover_mapping_type: str,
        scenario_and_id_folder: Path,
        flood_model_output_id: int
    ) -> None:
        r"""
        Define paths and variables relating to serving a Wflow scenario.

        Parameters
        ----------
        hydromt_path : Path
            A directory to where all necessary files are stored to run wflow model
        polygons: gpd.GeoDataFrame | None
            Polygons that are used to change the landcover information.
            This polygon dataframe has 'landcover' column with new values
        landcover_mapping_type : str
            Name of landcover dataset - globcover or lcdb
        scenario_and_id_folder : Path
            Directory to the scenario folder name with ID
        flood_model_output_id : int
            The ID of the flood model scenario that has been created using this wflow model
        """
        self.hydromt_path = hydromt_path
        self.polygons = polygons
        self.landcover_mapping_type = landcover_mapping_type
        self.scenario_and_id_folder = scenario_and_id_folder
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
        catchment_file = self.scenario_and_id_folder / "hydrological_process/wflow_test_full/staticgeoms/basins.geojson"
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
        is_baseline = self.polygons is None
        landcover_file = find_landcover_file(
            self.hydromt_path, self.landcover_mapping_type, self.scenario_and_id_folder, is_baseline
        )

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


def get_hydrograph_csv(flood_model_id: int, injection_point_feature_id: str) -> str:
    """
    Query the database for hydrograph data for an injection point and format it as CSV.

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
    query = """
            SELECT datetimes, flows
            FROM injection_points
            WHERE model_output_id = %(model_output_id)s
              AND "FID" = %(FID)s
            """
    params = {
        "model_output_id": flood_model_id,
        "FID": injection_point_feature_id,
    }
    # Read the data with datetimes and flows each in a list within one row.
    engine = setup_environment.get_database()
    with engine.connect() as conn:
        wide_form_df = pd.read_sql(query, conn, params=params)

    # Correct types, columns as lists
    wide_form_df.datetimes = wide_form_df.datetimes.apply(literal_eval)
    wide_form_df.flows = wide_form_df.flows.apply(literal_eval)

    # Expand the lists into long form df
    long_df = wide_form_df.explode(["datetimes", "flows"]).drop_duplicates()
    # Correct datetimes type object -> pd.datetime
    db_date_format = "%Y-%m-%d %H:%M:%S"
    long_df.datetimes = pd.to_datetime(long_df.datetimes, format=db_date_format)

    # Format the column names ready for display
    long_df = long_df.rename(columns={"datetimes": "Time", "flows": "Flow"})

    # Format data frame into a Comma Separated Values (CSV) string
    buffer = StringIO()
    long_df.to_csv(buffer, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    csv_string = buffer.getvalue()
    return csv_string
