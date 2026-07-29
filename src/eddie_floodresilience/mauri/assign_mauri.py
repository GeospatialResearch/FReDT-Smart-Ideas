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

"""Assigns Mauri metrics to given landcovers. Used as a prototype for community engagement."""
import importlib
import logging
from pathlib import Path
from textwrap import dedent

import geopandas as gpd
import pandas as pd
import rioxarray as rxr
from sqlalchemy import Connection

from eddie import geoserver as gs
from eddie.digitaltwin.tables import check_table_exists
from eddie.digitaltwin.setup_environment import get_database
from src.eddie_floodresilience.raster import polygonize_raster

log = logging.getLogger(__name__)


def serve_landcover_with_mauri(
    workspace_name: str, landcover_raster_path: Path, catchment_poly: gpd.GeoDataFrame, flood_model_output_id: int
) -> None:
    """
    Process and serves the landcover ready for visualisation with Mauri.

    Parameters
    ----------
    workspace_name: str
        The GeoServer workspace to add new layers to.
    landcover_raster_path: Path
        The path to the landcover raster file.
    catchment_poly: gpd.GeoDataFrame
        A GeoDataFrame containing a polygon to clip the landcover raster to.
    flood_model_output_id: int
        The scenario which this landcover belongs to.
    """
    # Polygonize landcover for area of interest
    log.info(f"Converting landcover raster {landcover_raster_path} to polygons.")
    with rxr.open_rasterio(landcover_raster_path) as landcover_raster:
        # Get catchment geometry in matching CRS
        catchment = catchment_poly.to_crs(landcover_raster.rio.crs).geometry
        # Clip raster to catchment AOI
        clipped_landcover = landcover_raster.rio.clip(catchment)
        # Convert raster to polygons
        landcover_polygons = polygonize_raster(clipped_landcover, column_name="landcover")
    # Add landcover and Mauri mappings into the database
    engine = get_database()
    with engine.connect() as conn:
        add_mauri_mapping_to_db(conn)
        add_polygonized_landcover_to_db(conn, landcover_polygons, flood_model_output_id)
        # Serve the landcover, joined with Mauri mappings
        add_landcover_mauri_geoserver_view(conn, workspace_name)


def add_mauri_mapping_to_db(conn: Connection, force_overwrite: bool = False) -> None:
    """
    Add mapping from LCDB classes to Mauri values to the database.
    Allows for later joins for analysis and interactive visualisation.

    Parameters
    ----------
    conn: Connection
        The database connection.
    force_overwrite: bool = False
        If True, force overwrite of existing Mauri mapping table.
    """
    # Check if we need to add the table
    mauri_mapping_table_name = "lcdb_mauri_mapping"
    if (not force_overwrite) and check_table_exists(conn, mauri_mapping_table_name):
        # If we don't have to add the table because it already exists then we shouldn't
        # Force overwrite will add it again anyway
        return
    log.info("Adding mapping between LCDB classes and Mauri values to database.")
    # Import the CSV in a way that allows it to be shared in package managers
    mauri_mapping_path = importlib.resources.files(f"{__package__}.resources") / "lcdb_mauri_mapping.csv"

    # Use pandas to add the mapping to the database.
    mauri_mapping_df = pd.read_csv(mauri_mapping_path, index_col="LCDB class")
    mauri_mapping_df.to_sql(mauri_mapping_table_name, conn, if_exists="replace")


def add_landcover_mauri_geoserver_view(conn: Connection, workspace_name: str) -> None:
    """
    Add a geoserver layer "lcdb_mauri_view" which joins the landcover and mauri tables for visualisation and analysis.
    If the layer is already present, then does nothing.

    Parameters
    ----------
    conn: Connection
        The connection to the database.
    workspace_name: str
        The name of the GeoServer workspace to add the layer to.
    """
    gs_layer_name = "lcdb_mauri_view"
    log.info(f"Adding GeoServer view {gs_layer_name} for landcover joined with Mauri mappings.")
    # Create the database geoserver store
    data_store = gs.create_main_db_store(workspace_name)
    landcover_sql_query = dedent(
        # @formatter:off - formatter impacts the %scenario% term
        """
        SELECT lcdb_mauri_mapping.*,
               landcover.flood_model_output_id,
               landcover.geometry
        FROM landcover
                 JOIN lcdb_mauri_mapping
                      ON landcover.landcover = lcdb_mauri_mapping."LCDB class"
        WHERE flood_model_output_id = %scenario%
        """
        # @formatter:on
    )

    landcover_mauri_xml_query = gs.database_layers.generate_metadata_elem(gs_layer_name, landcover_sql_query)
    gs.create_datastore_layer(
        conn, workspace_name, data_store, gs_layer_name, landcover_mauri_xml_query
    )


def add_polygonized_landcover_to_db(
    conn: Connection, landcover_gdf: gpd.GeoDataFrame, flood_model_output_id: int
) -> None:
    """
    Add the landcover polygons dataset to the database.

    Parameters
    ----------
    conn: Connection
        The database connection.
    landcover_gdf: gpd.GeoDataFrame
        The polygons of landcover types to add to the database.
    flood_model_output_id: int
        The id of the flood model output.
    """
    log.info("Adding polygonised landcover polygons to database.")
    landcover_gdf["flood_model_output_id"] = flood_model_output_id
    landcover_gdf.to_postgis("landcover", conn, if_exists="append")
