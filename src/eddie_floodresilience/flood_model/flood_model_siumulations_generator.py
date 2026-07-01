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

"""Abstract base class to run a flood model"""

from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
import logging

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from eddie import geoserver
from eddie.digitaltwin import setup_environment
from eddie.digitaltwin.utils import LogLevel, setup_logging

from src.eddie_floodresilience.config import EnvVariable
from src.eddie_floodresilience.flood_model.bg_flood_model import store_model_output_metadata_to_db
from src.eddie_floodresilience.flood_model.flooded_buildings import (
    find_flooded_buildings, store_flooded_buildings_in_database)
from . import serve_model
from .flood_model_inputs_generator import InjectionPointsFloodModelGenerator, TerrainGenerator

setup_logging(LogLevel.DEBUG)
log = logging.getLogger(__name__)


class BaseFloodModelSimulationsGenerator(ABC):
    """This class is to generate flood model simulations"""  # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        flood_model_path: Path,
        catchment_model_path: Path,
        hydromt_path: Path,
        river_name: str,
        precipitation_path: Path,
        aoi_boundary: list,
        adjust_manning: bool,
        start_time: datetime,
        end_time: datetime,
        crs: int = 2193,
        polygons: str = None,
        vectors: str = None
    ) -> None:
        """
        Generate flood model simulations

        Parameters
        ----------
        flood_model_path : Path
            Directory to folder storing terrain data
        catchment_model_path : Path
            Directory to folder storing catchment model results
        hydromt_path : Path
            A directory to where all necessary files are stored to run wflow model
        river_name: str
            Name of directory to where the river information files are stored
        precipitation_path : Path
            Directory to folder storing precipitation data
        aoi_boundary : list
            Boundaries' coordinates of area of interest.
            Format is [xmin, ymin, xmax, ymax]
        adjust_manning : bool
            True means adjusting Manning's n by resampling 4m Manning's n
            False means no Mannning's n adjustment
        start_time: datetime
            Starting time of the flood event
        end_time: datetime
            Ending time of the flood event
        crs : int = 2193
            Targeted crs. The default is 2193 for NZTM.
        polygons : str = None
            Name of polygon file that is used to change the landcover information.
            This polygon dataframe has 'landcover' column with new values
        vectors : str = None
            Name of vector file that is used to change the elevation information.
            This vector dataframe has 'value' column to specify increasing or decreasing elevation,
            and 'distance' column to specify how smooth to decrease elevation.
        """
        self.flood_model_path = flood_model_path
        self.catchment_model_path = catchment_model_path
        self.hydromt_path = hydromt_path
        self.river_name = river_name
        self.precipitation_path = precipitation_path
        self.aoi_boundary = aoi_boundary
        self.adjust_manning = adjust_manning
        self.start_time = start_time
        self.end_time = end_time
        self.crs = crs
        self.polygons = polygons
        self.vectors = vectors

        # Call out class to generate common terrain data
        self.terrain = TerrainGenerator(
            self.flood_model_path,
            self.hydromt_path,
            self.river_name,
            self.aoi_boundary,
            self.polygons,
            self.vectors,
            self.crs
        )

        # Generate common terrain data
        self.terrain_bounding_box, self.terrain_crs_clipped = self.terrain.terrain_data_generator()

    def injection_points_for_flood_model_generator(self) -> None:
        """Generate injection points for flood model"""
        # Call out class used to generate injection points for flood model
        injection_points_for_flood_model = InjectionPointsFloodModelGenerator(
            self.flood_model_path,
            self.catchment_model_path,
            self.terrain_bounding_box,
            self.start_time,
            self.end_time,
            self.polygons,
            self.crs
        )

        # Generate injection points for flood model
        injection_points_for_flood_model.injection_points_flow_generator()

    @abstractmethod
    def precipitation_data_for_flood_model_generator(self) -> None:
        """Generate precipitation data for flood model"""

    @abstractmethod
    def parameter_files_for_flood_model_generator(self) -> Path:
        """
        Generate parameters files for flood model

        Return
        ------
        Path
            The path to the output directory generated for these parameters
        """

    @abstractmethod
    def flood_model_simulations_generator(self, output_dir: Path) -> Path:
        """
        Generate flood simulations by running flood model

        Parameters
        ----------
        output_dir : Path
            The path to the output directory, to allow for serving.

        Returns
        -------
        Path
            The Flood Model maximum extents raster file
        """

    @abstractmethod
    def flood_model_executor(self) -> int:
        """
        Generate necessary inputs for flood model

        Returns
        -------
        int
            The Flood Model output ID
        """

    def serve_flood_model_outputs(self, output_tif: Path) -> int:
        """
        Add max flood model output data to database and geoserver for serving.

        Parameters
        ----------
        output_tif : Path
            The output of the flood model output.

        Returns
        -------
        int
            The flood model output ID.
            Returns -1 if GeoServer is disabled for testing.
        """
        if not EnvVariable.IS_GEOSERVER_ACTIVE:
            return -1

        # Retrieve the AOI as a GeoDataFrame
        bbox_gdf = gpd.GeoDataFrame(geometry=[box(*self.aoi_boundary)], crs="EPSG:2193")

        # Store metadata related to the BG Flood model output in the database
        engine = setup_environment.get_database()
        with engine.connect() as conn:
            model_output_id = store_model_output_metadata_to_db(conn, output_tif, bbox_gdf)
            # Find buildings that are flooded to a depth greater than or equal to 0.1m
            log.info("Analysing flooded buildings")
            flooded_buildings = find_flooded_buildings(conn, bbox_gdf, output_tif,
                                                       flood_depth_threshold=0.03)
            log.info("Analysed flooded buildings - adding flooded buildings to database")
            store_flooded_buildings_in_database(conn, flooded_buildings, model_output_id)

        # Assign a new workspace name based on the db_name, to prevent name clashes if running multiple databases
        db_name = EnvVariable.POSTGRES_DB
        workspace_name = f"{db_name}-dt-model-outputs"
        geoserver.create_workspace_if_not_exists(workspace_name)
        # Add the gtiff to geoserver
        layer_name = f"output_{model_output_id}"
        geoserver.add_gtiff_to_geoserver(output_tif, workspace_name, layer_name)
        serve_model.create_viridis_style_if_not_exists()

        return model_output_id

    def serve_injection_points(self, model_output_id: int) -> None:
        # Read injection point files
        flow_df_path = self.flood_model_path / "injection_points_flow.csv"
        flow_df = pd.read_csv(flow_df_path)
        points_path = self.flood_model_path / "injection_points.shp"
        injection_points = gpd.read_file(points_path)

        # Transform flow_df into one-row-per-point structure, matching `injection_points`.
        flow_transformed = flow_df.T
        # Name columns based on timestamp
        flow_transformed.columns = flow_transformed.iloc[0]
        # Drop timestamp row
        flow_transformed = flow_transformed.iloc[1:]
        # Match the index name to its true representation
        flow_transformed = flow_transformed.rename_axis("FID")

        # Merge the two datasets, so geometry and flow data are in one
        flow_points = injection_points.merge(flow_transformed, on="FID").copy(deep=True)

        # Further transform into narrow form, with one column listing timestamps and one column listing flows,
        # while still retaining one-row-per-point.
        # This is the form expected for visualization in the front-end.

        static_cols = ["geometry", "FID"]
        time_cols = [col for col in flow_points.columns if col not in static_cols]
        narrow_df = flow_points[static_cols]
        # Correctly set the geomery, since it may not be missing
        narrow_df = narrow_df.set_geometry("geometry")
        # Add the full list of datetimes to each row
        narrow_df["datetimes"] = [time_cols] * len(narrow_df)
        # Add a list of flows to each row, parallel to the datetimes.
        narrow_df["flows"] = flow_points[time_cols].values.tolist()

        # Add flood_model_id so we can select by it in database queries.
        narrow_df["model_output_id"] = model_output_id

        # Set up geoserver workspace and store
        db_name = EnvVariable.POSTGRES_DB
        workspace_name = f"{db_name}-flood-model-inputs"
        geoserver.create_workspace_if_not_exists(workspace_name)
        store_name = geoserver.create_main_db_store(workspace_name)

        # Append the hydrograph data to the database and serve it
        engine = setup_environment.get_database()
        with engine.connect() as conn:
            # Append the data to the database
            table_name = "injection_points"
            narrow_df.to_postgis(table_name, conn, if_exists="append", index=False)
            # Serve the data
            geoserver.create_datastore_layer(conn, workspace_name, store_name, table_name)
