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

import json
import logging
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import geopandas as gpd
import xarray as xr

from .wflow_data_catalog_generator import DataCatalogGenerator
from .wflow_build_generator import WflowBuildGenerator
from .wflow_groundwater_parameters_generator import GroundwaterParameterGenerator
from ..solutions.nature import lai_modification
from ..solutions.nature.landcover import LandcoverClassDataset
from ..solutions.nature.lai_modification import LAIModification
from ..solutions.nature.kvfrac_modification import KvfracModification

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
    landcover : LandcoverClassDataset = LandcoverClassDataset.GLOBCOVER = 'globcover'
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
        landcover: LandcoverClassDataset = LandcoverClassDataset.GLOBCOVER
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
        landcover : LandcoverClassDataset = LandcoverClassDataset.GLOBCOVER = 'globcover'
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

        # Load river-specific parameters from json
        self.river_params = self._load_river_params()

    def _load_river_params(self) -> dict:
        """
        Load river-specific parameters from
        <hydromt_path>/river_data/<river_name>/<river_name>.json

        Returns
        -------
        dict
            Dictionary of river-specific parameters
        """
        json_path = Path(self.hydromt_path) / "river_data" / self.river_name / f"{self.river_name}.json"
        log.info(f"Loading river parameters from {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

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
                "hydromt", "update", "wflow_sbm",
                str(self.scenario_and_id_folder.parent / r"original_scenario/hydrological_process/wflow_test_full"),
                "-o", str(output_folder_name),
                "-i", str(wflow_build_file),
                "-d", str(data_catalog_file),
                "-vv"
            ]

        else:

            # Set up command for preprocessing
            preprocessing_command_list = [
                "hydromt", "build", "wflow_sbm",
                str(output_folder_name),
                # "-r", region_information,
                "-i", str(wflow_build_file),
                "-d", str(data_catalog_file),
                "-vv"
            ]

        # Preprocess data
        subprocess.run(
            preprocessing_command_list,
            check=True
        )

    def groundwater_parameters_command(self):
        """Set up groundwater parameters for wflow simulation"""
        # Read groundwater parameters from json
        gw_pars = self.river_params["setup_groundwater_pars"]

        groundwater_paras = GroundwaterParameterGenerator(
            path=self.hydrological_process_path / "wflow_test_full",
            conductivity_divisor=50.0,
            riverbed_thickness_scaled=60.0,
            b_exponent=0.5,
            min_depth=1.0,
            max_depth=2.5,
            f_multiplier=gw_pars["f_multiplier"],
            gwf_f_multiplier=gw_pars["gwf_f_multiplier"]
        )

        # Generate groundwater parameters
        groundwater_paras.run()

    def lai_modification_command(self) -> None:
        """Modify LAI data according to the nature-based solution scenario"""
        # Set up settings to modify LAI data
        lai_modification = LAIModification(
            staticmaps_original=self.scenario_and_id_folder.parent / "original_scenario" / "hydrological_process" / "wflow_test_full" / "staticmaps.nc",
            staticmaps_scenario=self.hydrological_process_path / "wflow_test_full" / "staticmaps.nc"
        )

        # Modify LAI data
        lai_modification.run()

    def kvfrac_modification_command(self, scenario_kvfrac: tuple) -> None:
        """Modify kvfrac data according to the nature-based solution scenario"""
        # Set up settings to modify kvfrac data
        kvfrac_modification = KvfracModification(
            staticmaps_original=self.scenario_and_id_folder.parent / "original_scenario" / "hydrological_process" / "wflow_test_full" / "staticmaps.nc",
            staticmaps_scenario=self.hydrological_process_path / "wflow_test_full" / "staticmaps.nc",
            polygons=self.polygons,
            scenario_kvfrac=scenario_kvfrac
        )

        # Modify kvfrac data
        kvfrac_modification.run()

    def conductance_modification_command(self, infilt_multiplier: float, exfilt_multiplier: float) -> None:
        """Multiply infilt_cond and exfilt_cond inside the polygons (from the original values)"""
        orig = self.scenario_and_id_folder.parent / "original_scenario" / "hydrological_process" / "wflow_test_full" / "staticmaps.nc"
        scen = self.hydrological_process_path / "wflow_test_full" / "staticmaps.nc"

        with xr.open_dataset(orig) as ds:
            ds_orig = ds.load()
        with xr.open_dataset(scen) as ds:
            ds_scen = ds.load()

        # Same polygon mask as kvfrac (inside polygons and inside catchment)
        mask = KvfracModification(scen, self.polygons, ()).build_modify_mask(ds_scen)
        mask = xr.DataArray(mask, coords={"y": ds_scen["y"], "x": ds_scen["x"]}, dims=("y", "x"))

        # Multiply inside polygons, keep original values elsewhere
        ds_scen["infilt_cond"] = ds_orig["infilt_cond"].where(~mask, ds_orig["infilt_cond"] * infilt_multiplier)
        ds_scen["exfilt_cond"] = ds_orig["exfilt_cond"].where(~mask, ds_orig["exfilt_cond"] * exfilt_multiplier)

        ds_scen.to_netcdf(scen, mode="w")

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
        # """Generate wflow model simulation"""
        # # Generate files for preprocessing
        # self.files_for_preprocessing_generator()
        #
        # # Preprocess data for wflow model
        # self.preprocessing_command()

        # Fix diluted LAI for converted land cover (nature scenarios only)
        if self.polygons is not None:
            # Adjust LAI data
            self.lai_modification_command()

            # Read kvfrac from the original scenario's staticmaps
            orig = self.scenario_and_id_folder.parent / "original_scenario" / "hydrological_process" / "wflow_test_full" / "staticmaps.nc"
            with xr.open_dataset(orig) as ds_orig:
                kvfrac = float(ds_orig["kvfrac"].isel(layer=0).median())

            # At the moment, this is only for forest and wetland
            # In the future, this should be created as a column in land cover mapping and can be changed
            if (self.polygons["landcover"] == 45).any():
                # This is for wetland
                kvfrac = kvfrac + 0.007
                self.conductance_modification_command(infilt_multiplier=4, exfilt_multiplier=1/2)
            else:
                # This is for forest
                kvfrac = kvfrac + 0.01

            # Adjust infiltration of runoff
            self.kvfrac_modification_command(scenario_kvfrac=(kvfrac, 1.0, 1.0, 1.0, 1.0, 1.0))

        else:
            # Generate groundwater parameters (before kvfrac, so it cannot overwrite it)
            self.groundwater_parameters_command()

        # Generate wflow model simulation
        self.simulation_command()