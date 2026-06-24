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

"""Runs LISFLOOD-FP flood model"""
# pylint: disable=duplicate-code

from pathlib import Path
from datetime import datetime
import logging
import platform
import subprocess

import geopandas as gpd
from shapely.geometry import box

from eddie import geoserver
from eddie.digitaltwin import setup_environment
from eddie.digitaltwin.utils import LogLevel, setup_logging

from src.eddie_floodresilience.config import EnvVariable
from src.eddie_floodresilience.flood_model.bg_flood_model import store_model_output_metadata_to_db
from src.eddie_floodresilience.flood_model.flooded_buildings import (
    find_flooded_buildings, store_flooded_buildings_in_database)
from .. import serve_model
from ..flood_model_inputs_generator import InjectionPointsFloodModelGenerator, TerrainGenerator
from .lisflood_inputs_generator import TerrainFloodModelGenerator
from .lisflood_parameters_generator import LisfloodParametersGenerator
from .lisflood_precipitation import LisfloodPrecipitationGenerator, LisfloodPrecipitationFloodModelGenerator

setup_logging(LogLevel.DEBUG)
log = logging.getLogger(__name__)


class LisFloodModelSimulationsGenerator:
    """This class is to generate flood model simulations."""  # pylint: disable=too-many-instance-attributes

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

    def terrain_data_for_flood_model_generator(self) -> None:
        """Generate terrain data for flood model (LISFLOOD-FP)"""
        # Call out class used to generate terrain data for flood model
        terrain_data_for_flood_model = TerrainFloodModelGenerator(
            self.flood_model_path,
            self.hydromt_path,
            self.river_name,
            self.terrain_crs_clipped,
            self.adjust_manning,
            self.crs
        )

        # Generate terrain data for flood model
        terrain_data_for_flood_model.execute_terrain_data_generator()

    def injection_points_for_flood_model_generator(self) -> None:
        """Generate injection points for flood model (LISFLOOD-FP)"""
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

    def precipitation_data_for_flood_model_generator(self) -> None:
        """Generate precipitation data for flood model"""
        # Call out class used to generate precipitation data
        precipitation_generator = LisfloodPrecipitationGenerator(
            self.flood_model_path,
            self.precipitation_path,
            self.terrain_bounding_box,
            self.start_time,
            self.end_time,
            self.crs
        )

        # Generate precipitation data
        precipitation_data = precipitation_generator.precipitation_data_generator()

        # Call out class used to generate precipitation data for flood model
        precipitation_data_for_flood_model = LisfloodPrecipitationFloodModelGenerator(
            self.flood_model_path,
            precipitation_data
        )

        # Generate precipitation data for flood model
        precipitation_data_for_flood_model.precipitation_for_flood_model_generator()

    def parameters_files_for_flood_model_generator(self) -> Path:
        """
        Generate parameters files for flood model

        Return
        ------
        Path
            The path to the output directory generated for these parameters
        """
        # Call out class used to generate parameter files
        parameters_files_generator = LisfloodParametersGenerator(
            self.flood_model_path,
            self.terrain_bounding_box,
            self.start_time,
            self.end_time,
            self.polygons,
            self.vectors
        )

        # Generate parameter files
        output_dir = parameters_files_generator.parameter_files_generator()
        return output_dir

    def flood_model_simulations_generator(self, output_dir: Path) -> int:
        """
        Generate flood simulations by running flood model

        Parameters
        ----------
        output_dir : Path
            The path to the output directory, to allow for serving.

        Returns
        -------
        int
            The Flood Model output ID
        """
        # Set up path to log file
        log_file_path = self.flood_model_path / "simulation_log.log"

        # Set up path to parameters' file
        par_file_path = str(self.flood_model_path / "par.par")

        # Identify the LISFLOOD-FP executable, accounting for OS differences
        operating_system = platform.system()
        linux_path = EnvVariable.HYDROMT_PATH / "lisflood"
        match operating_system:
            case "Windows":
                lisflood_path = EnvVariable.HYDROMT_PATH / "lisflood_v8_1_0.exe"
            case "Linux":
                lisflood_path = linux_path
            case _:
                lisflood_path = linux_path
                log.warning(
                    f"{operating_system} is not officially supported. Only Windows and Linux are officially supported.")
                log.warning(f"Attempting to run LISFLOOD-FP linux script in {operating_system}")

        # Flood simulation command
        flood_simulation_command = [
            lisflood_path,
            "-v",
            par_file_path
        ]

        # Generate flood model simulations
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            log.info("Running LISFLOOD-FP flood simulation")
            subprocess.run(
                flood_simulation_command,
                stdout=log_file,
                stderr=subprocess.STDOUT,  # add error into log file if appears
                check=True
            )

        model_output_id = self.serve_flood_model_outputs(output_dir)
        return model_output_id

    def flood_model_executor(self) -> int:
        """
        Generate necessary inputs for flood model

        Returns
        -------
        int
            The Flood Model output ID
        """
        # Four cases:
        # 1. Original scenario (polygon=None, vector=None)
        # 2. Polygon=None, vector!=None
        # 3. Polygon!=None, vector=None
        # 4. Polygon!=None, vector!=None
        # This 'if' includes 1
        if self.polygons is None and self.vectors is None:
            # Generate terrain data for flood model
            self.terrain_data_for_flood_model_generator()

            # Generate injection points for flood model
            self.injection_points_for_flood_model_generator()

            # Generate precipitation data for flood model
            # self.precipitation_data_for_flood_model_generator()

            # Generate parameter files for flood model
            output_dir = self.parameters_files_for_flood_model_generator()

            # Generate simulations by running flood model
            model_output_id = self.flood_model_simulations_generator(output_dir)

        # This 'elif' includes 3 and 4
        elif self.polygons is not None or self.vectors is None:
            # Generate injection points for flood model
            self.injection_points_for_flood_model_generator()

            # Generate parameter files for flood model
            output_dir = self.parameters_files_for_flood_model_generator()

            # Generate simulations by running flood model
            model_output_id = self.flood_model_simulations_generator(output_dir)

        # This 'else' includes 2
        else:
            # Generate terrain data for flood model
            self.terrain_data_for_flood_model_generator()

            # Generate parameter files for flood model
            output_dir = self.parameters_files_for_flood_model_generator()

            # Generate simulations by running flood model
            model_output_id = self.flood_model_simulations_generator(output_dir)
        return model_output_id

    def serve_flood_model_outputs(self, output_directory: Path) -> int:
        """
        Add max flood model output data to database and geoserver for serving.

        Parameters
        ----------
        output_directory : Path
            The output directory for the flood model output.

        Returns
        -------
        int
            The flood model output ID.
            Returns -1 if GeoServer is disabled for testing.
        """
        if not EnvVariable.IS_GEOSERVER_ACTIVE:
            return -1

        # Convert the ASCII raster to GeoTIFF
        max_asc = output_directory / "out.max"
        time = datetime.now().strftime("%Y%m%d%H%M%S")
        max_gtiff = output_directory / f"{output_directory.name}-{time}-out.tif"
        serve_model.asc_to_gtiff(max_asc, max_gtiff)
        # Retrieve the AOI as a GeoDataFrame
        bbox_gdf = gpd.GeoDataFrame(geometry=[box(*self.aoi_boundary)], crs="EPSG:2193")

        # Store metadata related to the BG Flood model output in the database
        engine = setup_environment.get_database()
        with engine.connect() as conn:
            model_output_id = store_model_output_metadata_to_db(conn, max_gtiff, bbox_gdf)
            # Find buildings that are flooded to a depth greater than or equal to 0.1m
            log.info("Analysing flooded buildings")
            flooded_buildings = find_flooded_buildings(conn, bbox_gdf, max_gtiff,
                                                       flood_depth_threshold=0.03)
            log.info("Analysed flooded buildings - adding flooded buildings to database")
            store_flooded_buildings_in_database(conn, flooded_buildings, model_output_id)

        # Assign a new workspace name based on the db_name, to prevent name clashes if running multiple databases
        db_name = EnvVariable.POSTGRES_DB
        workspace_name = f"{db_name}-dt-model-outputs"
        geoserver.create_workspace_if_not_exists(workspace_name)
        # Add the gtiff to geoserver
        layer_name = f"output_{model_output_id}"
        geoserver.add_gtiff_to_geoserver(max_gtiff, workspace_name, layer_name)
        serve_model.create_viridis_style_if_not_exists()

        return model_output_id
