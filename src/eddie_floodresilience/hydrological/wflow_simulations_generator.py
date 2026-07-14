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

"""Runs Wflow model simulations"""

import logging
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import geopandas as gpd

from eddie.digitaltwin.utils import setup_logging, LogLevel
from .wflow_data_catalog_generator import DataCatalogGenerator
from .wflow_build_generator import WflowBuildGenerator

setup_logging(LogLevel.DEBUG)
log = logging.getLogger(__name__)


class WflowSimulationsGenerator:
    """
    This class is to generate wflow model simulations

    Attributes
    ----------
    hydromt_path: Path
        A directory to where all necessary files are stored to run wflow model
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
    scenario_and_id_folder : Path
        Directory to the scenario folder name with ID
    polygons : gpd.GeoDataFrame | None = None
        Polygons that are used to change the landcover information.
        This polygon dataframe has 'landcover' column with new values
    resolution : float
        Resolution for flow data.
        Default is 0.00045 (in crs 4326) ~ 50 m (in crs 2193)
    landcover : str = 'globcover'
        Name of land cover dataset. Default is 'globcover'
    """  # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        hydromt_path: Path,
        river_name: str,
        forcing_path: Path,
        start_time: datetime,
        end_time: datetime,
        flood_aoi_boundary: list,
        num_threads: int,
        scenario_and_id_folder: Path,
        polygons: gpd.GeoDataFrame | None = None,
        resolution: float = 0.00045,
        landcover: str = 'globcover'
    ) -> None:
        """
        Generate wflow model simulations

        Parameters
        ----------
        hydromt_path: Path
            A directory to where all necessary files are stored to run wflow model
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
        scenario_and_id_folder : Path
            Directory to the new scenario ID for scenario
        polygons : gpd.GeoDataFrame | None = None
            Polygons that are used to change the landcover information.
            This polygon dataframe has 'landcover' column with new values
        resolution : float
            Resolution for flow data.
            Default is 0.00045 (in crs 4326) ~ 50 m (in crs 2193)
        landcover : str = 'globcover'
            Name of land cover dataset. Default is 'globcover'
        """
        self.hydromt_path = hydromt_path
        self.river_name = river_name
        self.forcing_path = forcing_path
        self.start_time = start_time
        self.end_time = end_time

        self.flood_aoi_boundary = flood_aoi_boundary

        self.num_threads = num_threads
        self.scenario_and_id_folder = scenario_and_id_folder
        # Set up hydrological process path
        self.hydrological_process_path = self.scenario_and_id_folder / "hydrological_process"

        self.polygons = polygons
        self.resolution = resolution
        self.landcover = landcover

    def files_for_preprocessing_generator(self) -> None:
        """Generate files for preprocessing"""
        # Generate data_catalog.yml
        data_catalog = DataCatalogGenerator(
            self.hydromt_path,
            self.forcing_path,
            self.river_name,
            self.scenario_and_id_folder,
            self.polygons,
            self.landcover
        )
        data_catalog.data_catalog_generator()

        # Generate wflow_build.yml
        wflow_build = WflowBuildGenerator(
            self.start_time,
            self.end_time,
            self.resolution,
            self.hydromt_path,
            self.river_name,
            self.forcing_path,
            self.scenario_and_id_folder,
            self.polygons,
            self.landcover
        )
        wflow_build.wflow_build_generator()

    def preprocessing_command(self) -> None:
        """Set up preprocessing command and preprocess data for wflow model"""
        log.info("Preprocessing data for wflow model")

        # Wflow result folder
        output_folder_name = self.hydrological_process_path / "wflow_test_full"

        # Find wflow build file
        wflow_build_file = self.hydrological_process_path / "wflow_build.yml"

        # Find data catalog file
        data_catalog_file = self.hydrological_process_path / "data_catalog.yml"

        if self.polygons is not None:
            # Set up command
            preprocessing_command_list = [
                "hydromt", "update", "wflow",
                str(self.scenario_and_id_folder.parent / r"original_scenario/hydrological_process/wflow_test_full"),
                "-o", str(output_folder_name),
                "-i", str(wflow_build_file),
                "-d", str(data_catalog_file),
                "-vv"
            ]

        else:
            # Get subbasin river outlet
            subbasin_river_outlet = gpd.read_file(
                self.scenario_and_id_folder.parent / 'terrain/river_outlet.shp'
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

            # Force override is turned off for performance, so we must manually remove the run_default dir if required.
            run_default_dir = output_folder_name / "run_default"
            if run_default_dir.exists():
                shutil.rmtree(run_default_dir)

            # Set up command for preprocessing
            preprocessing_command_list = [
                "hydromt", "build", "wflow",
                str(output_folder_name),
                "-r", region_information,
                "-i", str(wflow_build_file),
                "-d", str(data_catalog_file),
                "-vv"
            ]

        # Preprocess data
        subprocess.run(
            preprocessing_command_list,
            check=True
        )

    def simulation_command(self) -> None:
        """Set up simulation command and generate simulation"""
        # Set up output folder name
        output_folder_name = self.hydrological_process_path / "wflow_test_full"

        # Create folder name
        output_log = "wflow_run.log"

        log.info("Running wflow simulation.")
        # Set up simulation command
        simulation_command = [
            "julia",
            "-t", str(self.num_threads),
            "-e",
            f'cd("{output_folder_name.as_posix()}"); using Wflow; Wflow.run("wflow_sbm.toml")'
        ]

        # Run the command and write output to log
        with open(output_folder_name / output_log, "w", encoding="utf-8") as f:
            subprocess.check_call(
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
