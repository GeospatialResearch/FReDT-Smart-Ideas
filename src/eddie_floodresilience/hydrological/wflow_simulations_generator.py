# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 08:43:08 2026

@author: mng42
"""

from pathlib import Path
from datetime import datetime
import geopandas as gpd

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
        flood_aoi_boundary: list,
        num_threads: int,
        polygons: str = None,
        resolution: float = 0.00045,
        landcover: str = 'globcover'
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
        flood_aoi_boundary : list
            Boundaries' coordinates of area of interest.
            Format is [xmin, ymin, xmax, ymax]
        num_threads : int
            Number of threads that controls how fast the wflow model can run
        polygons : str = None
            Name of polygon file that is used to change the landcover information.
            This polygon dataframe has 'landcover' column with new values
        resolution : float
            Resolution for flow data. 
            Default is 0.00045 (in crs 4326) ~ 50 m (in crs 2193)
        landcover : str = 'globcover'
            Name of land cover dataset. Default is 'globcover'
        """
        self.hydromt_path = hydromt_path
        self.wflow_model_path = wflow_model_path
        self.river_name = river_name
        self.forcing_path = forcing_path
        self.start_time = start_time
        self.end_time = end_time

        self.flood_aoi_boundary = flood_aoi_boundary

        self.num_threads = num_threads

        self.polygons = polygons
        self.resolution = resolution
        self.landcover = landcover
        
    def files_for_preprocessing_generator(self) -> None:
        """Generate files for preprocessing"""
        # Generate data_catalog.yml
        data_catalog = DataCatalogGenerator(
            self.hydromt_path,
            self.wflow_model_path,
            self.forcing_path,
            self.river_name,
            self.polygons,
            self.landcover
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
            self.forcing_path,
            self.polygons,
            self.landcover
        )
        wflow_build.wflow_build_generator()
        
    def preprocessing_command(self) -> None:
        """Set up preprocessing command and preprocess data for wflow model"""
        if self.polygons is not None:
            # Find existing file
            existing_file = sorted(
                self.wflow_model_path.glob("wflow_test_full_landcover_*")
            )

            # Set ID for file
            number = len(existing_file) + 1

            # Create folder name
            output_foldername = f"wflow_test_full_landcover_{number:03d}"

            # Find wflow build file
            wflow_build_file = str(max(
                Path(self.wflow_model_path).glob("wflow_build_landcover_*.yml"),
                default=Path(self.wflow_model_path) / "wflow_build_landcover_001.yml"
            ))

            # Find data catalog file
            data_catalog_file = str(max(
                Path(self.wflow_model_path).glob("data_catalog_landcover_*.yml"),
                default=Path(self.wflow_model_path) / "data_catalog_landcover_001.yml"
            ))

            preprocessing_command_list = [
                "hydromt", "update", "wflow",
                str(self.wflow_model_path / "wflow_test_full"),
                "-o", str(self.wflow_model_path / output_foldername),
                "-i", str(self.wflow_model_path / wflow_build_file),
                "-d", str(self.wflow_model_path / data_catalog_file),
                "-vv"
            ]

        else:
            # Get subbasin river outlet
            subbasin_river_outlet = gpd.read_file(
                self.wflow_model_path / 'river_outlet.shp'
            )

            # Get subbasin river outlet coordinates
            subbasin_river_outlet_coords = list(
                subbasin_river_outlet.geometry.iloc[0].coords
            )[0]

            # Make sure it is in list style
            subbasin_river_outlet_coords_list = [
                subbasin_river_outlet_coords[0], subbasin_river_outlet_coords[1]
            ]

            # Set up region information
            region_information = str({
                "subbasin": subbasin_river_outlet_coords_list,
                "strord": 4,
                "bbox": self.flood_aoi_boundary
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
        if self.polygons is not None:
            # Set up folder
            output_foldername = str(max(
                Path(self.wflow_model_path).glob("wflow_test_full_landcover_*"),
                default=Path(self.wflow_model_path) / "wflow_test_full_landcover_001"
            ))

            wflow_simulation_path = self.wflow_model_path / output_foldername

            # Set up log
            existing_log = sorted(
                self.wflow_model_path.glob("wflow_run_landcover_*.log")
            )

            # Set ID for file
            number = len(existing_log) + 1

            # Create folder name
            output_log = f"wflow_run_landcover_{number:03d}.log"

        else:
            wflow_simulation_path = self.wflow_model_path / "wflow_test_full"
            output_log = f"wflow_run.log"

        # Set up simulation command
        simulation_command = [
            "julia",
            "-t", str(self.num_threads),
            "-e",
            f'cd("{wflow_simulation_path.as_posix()}"); using Wflow; Wflow.run("wflow_sbm.toml")'
        ]

        # Run the command and write output to log
        with open(wflow_simulation_path / output_log, "w") as f:
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
