# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:12:29 2026

@author: mng42
"""

from pathlib import Path
from datetime import datetime

from .flood_model_inputs_generator import TerrainGenerator, \
                                          TerrainFloodModelGenerator, \
                                          InjectionPointsFloodModelGenerator
                            
from .flood_model_precipitation import PrecipitationGenerator, PrecipitationFloodModelGenerator

from .flood_model_parameters_generator import ParametersFloodModelGenerator

import subprocess


class FloodModelSimulationsGenerator():
    """This class is to generate flood model simulations"""
    
    def __init__(
        self,
        flood_model_path: Path,
        catchment_model_path: Path,
        precipitation_path: Path,
        aoi_boundary: list,
        start_time: datetime,
        end_time: datetime,
        crs: int = 2193,
        polygons: int = None,
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
        precipitation_path : Path
            Directory to folder storing precipitation data
        aoi_boundary : list
            Boundaries' coordinates of area of interest.
            Format is [xmin, ymin, xmax, ymax]
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
        self.precipitation_path = precipitation_path
        self.aoi_boundary = aoi_boundary
        self.start_time = start_time
        self.end_time = end_time
        self.crs = crs
        self.polygons = polygons
        self.vectors = vectors

        # Call out class to generate common terrain data
        self.terrain = TerrainGenerator(
            self.flood_model_path, 
            self.aoi_boundary,
            self.crs
        )
        
        # Generate common terrain data
        self.terrain_bounding_box, self.terrain_crs_clipped = self.terrain.terrain_data_generator()
        
    def terrain_data_for_flood_model_generator(self):
        """Generate terrain data for flood model (LISFLOOD-FP)"""
        # Call out class used to generate terrain data for flood model
        terrain_data_for_flood_model = TerrainFloodModelGenerator(
            self.flood_model_path,
            self.terrain_crs_clipped,
            self.crs
        )
        
        # Generate terrain data for flood model
        terrain_data_for_flood_model.execute_terrain_data_generator()
    
    def injection_points_for_flood_model_generator(self):
        """Generate injection points for flood model (LISFLOOD-FP)"""
        # Call out class used to generate injection points for flood model
        injection_points_for_flood_model = InjectionPointsFloodModelGenerator(
            self.flood_model_path,
            self.catchment_model_path,
            self.terrain_bounding_box,
            self.start_time,
            self.end_time,
            self.crs
        )
        
        # Generate injection points for flood model
        injection_points_for_flood_model.injection_points_flow_generator()
        
    def precipitation_data_for_flood_model_generator(self):
        """Generate precipitation data for flood model"""
        # Call out class used to generate precipitation data
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
        
    def parameters_files_for_flood_model_generator(self):
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
        parameters_files_generator.parameters_files_generator()
        
    def flood_model_simulations_generator(self):
        """Generate flood simulations by running flood model"""
        # Set up path to log file
        log_file_path = self.flood_model_path / "simulation_log.log"
        
        # Set up path to flood model exe
        # This will be changed into constant local folder in the future
        flood_model_exe_path = r"C:\Users\mng42\spyder_hydromt\digital_twin_codes\for_lisflood\lisflood_v8_1_0.exe"
        
        # Set up path to parameters' file
        par_file_path = str(self.flood_model_path / "par.par")
        
        # Flood simulation command
        flood_simulation_command = [
            flood_model_exe_path,
            "-v",
            par_file_path
        ]
        
        # Generate flood model simulations
        with open(log_file_path, "w") as log_file:
            subprocess.run(
                flood_simulation_command,
                stdout=log_file,
                stderr=subprocess.STDOUT,  # add error into log file if appears
                check=True
            )
        
    def flood_model_executor(self):
        """Generate necessary inputs for flood model"""
        if self.polygons is not None or self.vectors is None:
            # Generate terrain data for flood model
            self.terrain_data_for_flood_model_generator()

            # Generate injection points for flood model
            self.injection_points_for_flood_model_generator()

            # Generate precipitation data for flood model
            self.precipitation_data_for_flood_model_generator()

            # Generate parameter files for flood model
            self.parameters_files_for_flood_model_generator()

            # Generate simulations by running flood model
            self.flood_model_simulations_generator()

        else:
            # Generate terrain data for flood model
            self.terrain_data_for_flood_model_generator()

            # Generate parameter files for flood model
            self.parameters_files_for_flood_model_generator()

            # Generate simulations by running flood model
            self.flood_model_simulations_generator()
        