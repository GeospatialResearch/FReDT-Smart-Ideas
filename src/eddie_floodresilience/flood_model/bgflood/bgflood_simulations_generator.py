# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:12:29 2026

@author: mng42
"""

from pathlib import Path
from datetime import datetime
import shutil


from .bgflood_inputs_generator import TerrainGenerator, InjectionPointsFloodModelGenerator

from .bgflood_precipitation import PrecipitationGenerator, PrecipitationFloodModelGenerator

from .bgflood_parameters_generator import ParametersFloodModelGenerator

from .. import serve_model
from src.eddie_floodresilience.config import EnvVariable

import platform
import subprocess


class BGFloodModelSimulationsGenerator():
    """This class is to generate flood model simulations"""

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

    def injection_points_for_flood_model_generator(self):
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

    def precipitation_data_for_flood_model_generator(self):
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

    def parameter_data_for_flood_model_generator(self):
        """Generate paramter data for flood model"""
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

    def find_output_folder_path(self):
        """Find output folder path"""
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

    def flood_model_simulations_generator(self):
        """Generate flood simulations by running flood model"""
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
                print(f"{operating_system} is not officially supported. Only Windows and Linux are officially supported.")
                print(f"Attempting to run BG_Flood linux script in {operating_system}")

        # Copy executable into scenario folder
        output_executable = output_folder_path / flood_model_exe_path.name
        shutil.copyfile(
            flood_model_exe_path,
            output_executable
        )  # todo shutil.copyfile is used instead of copy2 because of some kind of mysterious linux permissions bug

        # BG flood command
        bg_flood_command = [
            output_executable
        ]
        # Run simulation
        print("Running BG Flood simulation")
        with open(log_file_path, "w") as log_file:
            subprocess.check_call(
                bg_flood_command,
                cwd=output_folder_path,
                stdout=log_file,
                stderr=log_file
            )
        self.serve_flood_model_outputs(output_folder_path)


    def serve_flood_model_outputs(self, output_directory: Path):
        model_output = output_directory / "output.nc"
        serve_model.add_model_output_to_geoserver(model_output, -1)

    def flood_model_executor(self):
        """Generate necessary inputs for flood model"""
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
            self.flood_model_simulations_generator()

        # This 'elif' includes 3 and 4
        elif self.polygons is not None or self.vectors is None:
            # Generate injection points for flood model
            self.injection_points_for_flood_model_generator()

            # Generate parameter files for flood model
            self.parameter_data_for_flood_model_generator()

            # Generate simulations by running flood model
            self.flood_model_simulations_generator()

        # This 'else' includes 2
        else:
            # Generate parameter files for flood model
            self.parameter_data_for_flood_model_generator()

            # Generate simulations by running flood model
            self.flood_model_simulations_generator()