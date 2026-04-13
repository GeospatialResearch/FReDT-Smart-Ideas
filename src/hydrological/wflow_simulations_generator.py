# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 08:43:08 2026

@author: mng42
"""
import os
os.chdir(r"C:\Users\mng42\spyder_hydromt\digital_twin_codes\for_both")

from pathlib import Path

from wflow_data_catalog_generator import DataCatalogGenerator
from wflow_build_generator import WflowBuildGenerator


import subprocess


class WflowSimulationsGenerator():
    """This class is to generate wflow model simulations"""
    
    def __init__(
        self,
        hydromt_path: Path,
        wflow_model_path: Path,
        precipitation_path: Path,
        start_time: str,
        end_time: str,
        subbasin: list,
        strord: int,
        bbox: list,
        num_threads: int,
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
        precipitation_path: Path
            A directory to where the preciptation files are stored
        start_time : str
            Starting time of simulation.
            This should include the spin-up time. 
            Normally, it is 1-year before the flood event.
        end_time : str
            Ending time of simulation
            This should include some periods of time after the flood event.
            Normally, it is about 2-3 months.
        subbasin : list
            Outlet coordinates
        strord : int
            Minimum stream order
        bbox : list
            Given bounding box coordinates that contains the subbasin coordinates
        num_threads : int
            Number of threads that controls how fast the wflow model can run
        resolution : float
            Resolution for flow data. 
            Default is 0.00045 (in crs 4326) ~ 50 m (in crs 2193)
        """
        self.hydromt_path = hydromt_path
        self.wflow_model_path = wflow_model_path
        self.precipitation_path = precipitation_path
        self.start_time = start_time
        self.end_time = end_time
        
        self.subbasin = subbasin
        self.strord = strord
        self.bbox = bbox

        self.num_threads = num_threads
        
        self.resolution = resolution
        
    def files_for_preprocessing_generator(self) -> None:
        """Generate files for preprocessing"""
        # Generate data_catalog.yml
        data_catalog = DataCatalogGenerator(
            self.hydromt_path,
            self.wflow_model_path,
            self.precipitation_path
        )
        data_catalog.data_catalog_generator()
        
        # Generate wflow_build.yml
        wflow_build = WflowBuildGenerator(
            self.start_time,
            self.end_time,
            self.resolution,
            self.wflow_model_path    
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
        
        # Change backslash style
        wflow_simulation_path = str(wflow_simulation_path).replace("\\", "/")
        
        # Set up simulation command
        simulation_command = [
            "julia",
            "-t", str(self.num_threads),
            "-e",
            f'cd("{wflow_simulation_path}"); using Wflow; Wflow.run("wflow_sbm.toml")'
        ]

        # Run the command and write output to log
        with open(fr"{wflow_simulation_path}\wflow_run.log", "w") as f:
            process = subprocess.run(
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
