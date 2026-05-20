# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 08:43:08 2026

@author: mng42
"""

from pathlib import Path
from datetime import datetime

from .wflow_data_catalog_generator import DataCatalogGenerator
from .wflow_build_generator import WflowBuildGenerator


import subprocess


class WflowSimulationsGenerator():
    """This class is to generate wflow model simulations"""
    
    def __init__(
        self,
        hydromt_path: Path,
        wflow_model_path: Path,
        river_name: str,
        forcing_path: Path,
        start_time: datetime,
        end_time: datetime,
        subbasin: list,
        strord: int,
        bbox: list,
        num_threads: int,
        polygons: str = None,
        resolution: float = 0.00045
    ) -> None:
        """
        Generate wflow model simulations
        
        Parameters
        ----------
        hydromt_path: Path
            A directory to where all necessary files are stored to run wflow model
        wflow_model_path: Path
            A directory to where the data_catalog.yml is stored and to run wflow model
        river_name: str
            Name of directory to where the river information files are stored
        forcing_path: Path
            A directory to where the forcing files are stored
        start_time : datetime
            Starting time of simulation.
            This should include the spin-up time. 
            Normally, it is 1-year before the flood event.
        end_time : datetime
            Ending time of simulation
            This should include some periods of time after the flood event.
            Normally, it is 12 hours or 1 day.
        subbasin : list
            Outlet coordinates
        strord : int
            Minimum stream order
        bbox : list
            Given bounding box coordinates that contains the subbasin coordinates
        num_threads : int
            Number of threads that controls how fast the wflow model can run
        polygons : str = None
            Name of polygon file that is used to change the landcover information.
            This polygon dataframe has 'landcover' column with new values
        resolution : float
            Resolution for flow data. 
            Default is 0.00045 (in crs 4326) ~ 50 m (in crs 2193)
        """
        self.hydromt_path = hydromt_path
        self.wflow_model_path = wflow_model_path
        self.river_name = river_name
        self.forcing_path = forcing_path
        self.start_time = start_time
        self.end_time = end_time
        
        self.subbasin = subbasin
        self.strord = strord
        self.bbox = bbox

        self.num_threads = num_threads

        self.polygons = polygons
        self.resolution = resolution
        
    def files_for_preprocessing_generator(self) -> None:
        """Generate files for preprocessing"""
        # Generate data_catalog.yml
        data_catalog = DataCatalogGenerator(
            self.hydromt_path,
            self.wflow_model_path,
            self.forcing_path,
            self.river_name,
            self.polygons
        )
        data_catalog.data_catalog_generator()
        
        # Generate wflow_build.yml
        wflow_build = WflowBuildGenerator(
            self.start_time,
            self.end_time,
            self.resolution,
            self.wflow_model_path,
            self.hydromt_path,
            self.river_name,
            self.forcing_path
        )
        wflow_build.wflow_build_generator()
        
    def preprocessing_command(self) -> None:
        """Set up preprocessing command and preprocess data for wflow model"""
        # Set up region information
        region_information = str({
            "subbasin": self.subbasin,
            "strord": self.strord,
            "bbox": self.bbox
        })

        # Set up command for preprocessing
        preprocessing_command_list = [
            "hydromt", "build", "wflow",
            f"{self.wflow_model_path}/wflow_test_full",
            "-r", region_information,
            "-i", f"{self.wflow_model_path}/wflow_build.yml",
            "-d", f"{self.wflow_model_path}/data_catalog.yml",
            "--fo",
            "-vv"
        ]

        # Preprocess data
        subprocess.run(
            preprocessing_command_list,
            check=True
        )
        
    def simulation_command(self) -> None:
        """Set up simulation command and generate simulation"""
        # Set up path to wflow simulation folder
        wflow_simulation_path = self.wflow_model_path / "wflow_test_full"
        
        # Set up simulation command
        simulation_command = [
            "julia",
            "-t", str(self.num_threads),
            "-e",
            f'cd("{wflow_simulation_path.as_posix()}"); using Wflow; Wflow.run("wflow_sbm.toml")'
        ]

        # Run the command and write output to log
        with open(wflow_simulation_path / "wflow_run.log", "w") as f:
            process = subprocess.check_call(
                simulation_command, 
                stdout=f, 
                stderr=subprocess.STDOUT, 
                text=True
            )
        
    def wflow_model_simulations_pipeline(self) -> None:
        """Generate wflow model simulation"""
        # Generate files for preprocessing
        self.files_for_preprocessing_generator()

        # Preprocessing data for wflow model
        self.preprocessing_command()
        
        # Generate wflow model simulation
        self.simulation_command()
