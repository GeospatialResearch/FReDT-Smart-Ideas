# -*- coding: utf-8 -*-
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

"""This script runs BG-Flood simulations."""
# pylint: disable=duplicate-code

import logging
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from eddie.digitaltwin import setup_environment
from eddie.digitaltwin.utils import LogLevel, setup_logging

from src.eddie_floodresilience.config import EnvVariable
from src.eddie_floodresilience.flood_model.bg_flood_model import store_model_output_metadata_to_db
from src.eddie_floodresilience.flood_model.flooded_buildings import (
    find_flooded_buildings, store_flooded_buildings_in_database)
from src.eddie_floodresilience.flood_model.serve_model import add_model_output_to_geoserver
from .bgflood_inputs_generator import TerrainGenerator, InjectionPointsFloodModelGenerator
from .bgflood_parameters_generator import ParametersFloodModelGenerator
from .bgflood_precipitation import PrecipitationGenerator, PrecipitationFloodModelGenerator

setup_logging(LogLevel.DEBUG)
log = logging.getLogger(__name__)


class BGFloodModelSimulationsGenerator():
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
        """Generate injection points for flood model (BG_Flood)"""
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
        # Path to precipitation file
        precipitation_file = (
            self.flood_model_path / "precipitation_dynamic.nc"
        )

        # Check if file already exists
        if precipitation_file.exists():
            pass
        else:
            precipitation_generator = PrecipitationGenerator(
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
            precipitation_data_for_flood_model = PrecipitationFloodModelGenerator(
                self.flood_model_path,
                precipitation_data
            )

            # Generate precipitation data for flood model
            precipitation_data_for_flood_model.precipitation_for_flood_model_generator()

    def parameter_data_for_flood_model_generator(self) -> None:
        """Generate parameters files for flood model"""
        # Call out class used to generate parameter files
        parameters_files_generator = ParametersFloodModelGenerator(
            self.flood_model_path,
            self.terrain_bounding_box,
            self.start_time,
            self.end_time,
            self.polygons,
            self.vectors
        )

        # Generate parameter files
        parameters_files_generator.parameter_files_generator()

    def find_output_folder_path(self) -> Path:
        """
        Find output folder path.

        Returns
        -------
        Path
            the output folder path.
        """
        # Output folder paht to store BG flood programme
        # Polygons for land cover solutions
        if self.polygons is not None and self.vectors is not None:
            output_folder_path = max(
                Path(self.flood_model_path).glob("output_landcover_elevation_*"),
                default=Path(self.flood_model_path) / "output_landcover_elevation_001"
            )
        elif self.polygons is not None:
            output_folder_path = max(
                Path(self.flood_model_path).glob("output_landcover_*"),
                default=Path(self.flood_model_path) / "output_landcover_001"
            )
        elif self.vectors is not None:
            output_folder_path = max(
                Path(self.flood_model_path).glob("output_elevation_*"),
                default=Path(self.flood_model_path) / "output_elevation_001"
            )
        else:
            output_folder_path = max(
                Path(self.flood_model_path).glob("output_*"),
                default=Path(self.flood_model_path) / "output"
            )

        return output_folder_path

    def flood_model_simulations_generator(self) -> int:
        """
        Generate flood simulations by running flood model

        Returns
        -------
        int
            The model output ID of the flood model run
        """
        # Set up path to log file
        log_file_path = self.flood_model_path / "simulation_log.log"

        # Get the output folder path
        output_folder_path = self.find_output_folder_path()

        # Identify the BG-Flood Model executable, accounting for OS differences
        operating_system = platform.system()
        match operating_system:
            case "Windows":
                flood_model_exe_path = EnvVariable.FLOOD_MODEL_DIR / "BG_flood.exe"
            case "Linux":
                flood_model_exe_path = EnvVariable.FLOOD_MODEL_DIR / "BG_Flood"
            case _:
                flood_model_exe_path = EnvVariable.FLOOD_MODEL_DIR / "BG_Flood"
                log.warning(
                    f"{operating_system} is not officially supported. Only Windows and Linux are officially supported.")
                log.warning(f"Attempting to run BG_Flood linux script in {operating_system}")

        # Copy executable into scenario folder
        output_executable = output_folder_path / flood_model_exe_path.name
        # shutil.copyfile is used instead of copy2 because of some kind of mysterious linux permissions bug.
        # https://github.com/GeospatialResearch/FReDT-Smart-Ideas/issues/83
        shutil.copyfile(
            flood_model_exe_path,
            output_executable
        )

        # BG flood command
        bg_flood_command = [
            output_executable
        ]
        # Run simulation
        log.info("Running BG Flood simulation")
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            subprocess.check_call(
                bg_flood_command,
                cwd=output_folder_path,
                stdout=log_file,
                stderr=log_file
            )
        model_output_id = self.serve_flood_model_outputs(output_folder_path)
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

        model_output = output_directory / "output.nc"
        # Retrieve the AOI as a GeoDataFrame
        bbox_gdf = gpd.GeoDataFrame(geometry=[box(*self.aoi_boundary)], crs="EPSG:2193")

        # Store metadata related to the BG Flood model output in the database
        engine = setup_environment.get_database()
        with engine.connect() as conn:
            model_output_id = store_model_output_metadata_to_db(conn, model_output, bbox_gdf)
            # Find buildings that are flooded to a depth greater than or equal to 0.1m
            log.info("Analysing flooded buildings")
            flooded_buildings = find_flooded_buildings(conn, bbox_gdf, model_output, flood_depth_threshold=0.1)
            log.info("Analysed flooded buildings - adding flooded buildings to database")
            store_flooded_buildings_in_database(conn, flooded_buildings, model_output_id)
        # Add the model output to GeoServer for visualization
        add_model_output_to_geoserver(model_output, model_output_id)
        return model_output_id

    def flood_model_executor(self) -> int:
        """
        Generate necessary inputs for flood model

        Returns
        -------
        int
            The model output ID of the flood model run
        """
        # Four cases:
        # 1. Original scenario (polygon=None, vector=None)
        # 2. Polygon=None, vector!=None
        # 3. Polygon!=None, vector=None
        # 4. Polygon!=None, vector!=None
        # This 'if' includes 1
        if self.polygons is None and self.vectors is None:
            # Generate injection points for flood model
            self.injection_points_for_flood_model_generator()

            # Generate parameter files for flood model
            self.parameter_data_for_flood_model_generator()

            # Generate precipitation data for flood model
            self.precipitation_data_for_flood_model_generator()

            # Generate simulations by running flood model
            model_output_id = self.flood_model_simulations_generator()

        # This 'elif' includes 3 and 4
        elif self.polygons is not None or self.vectors is None:
            # Generate injection points for flood model
            self.injection_points_for_flood_model_generator()

            # Generate parameter files for flood model
            self.parameter_data_for_flood_model_generator()

            # Generate simulations by running flood model
            model_output_id = self.flood_model_simulations_generator()

        # This 'else' includes 2
        else:
            # Generate parameter files for flood model
            self.parameter_data_for_flood_model_generator()

            # Generate simulations by running flood model
            model_output_id = self.flood_model_simulations_generator()
        return model_output_id
