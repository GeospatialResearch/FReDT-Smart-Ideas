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

import logging
import platform
import shutil
import subprocess
from pathlib import Path

from src.eddie_floodresilience.config import EnvVariable
from src.eddie_floodresilience.flood_model.serve_model import convert_nc_to_gtiff
from .bgflood_parameters_generator import BGFloodParametersGenerator
from .bgflood_precipitation import BGFloodPrecipitationGenerator, BGFloodPrecipitationFloodModelGenerator
from ..flood_model_parameters_generator import FloodType
from ..flood_model_siumulations_generator import BaseFloodModelSimulationsGenerator

log = logging.getLogger(__name__)


class BGFloodModelSimulationsGenerator(BaseFloodModelSimulationsGenerator):
    """This class is to generate flood model simulations"""  # pylint: disable=too-many-instance-attributes

    def precipitation_data_for_flood_model_generator(self) -> None:
        """Generate precipitation data for flood model"""  # pylint: disable=duplicate-code
        # Call out class used to generate precipitation data
        # Path to precipitation file
        precipitation_file = (
            self.flood_model_path / "precipitation_dynamic.nc"
        )

        # Check if file already exists
        if precipitation_file.exists():
            pass
        else:
            precipitation_generator = BGFloodPrecipitationGenerator(
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
            precipitation_data_for_flood_model = BGFloodPrecipitationFloodModelGenerator(
                self.flood_model_path,
                precipitation_data
            )

            # Generate precipitation data for flood model
            precipitation_data_for_flood_model.precipitation_for_flood_model_generator()

    def parameter_files_for_flood_model_generator(self) -> Path:
        """
        Generate parameters files for flood model.
        Returns
        -------
        Path
            The directory the parameter file was created in.
        """  # pylint: disable=duplicate-code
        # Call out class used to generate parameter files
        parameters_files_generator = BGFloodParametersGenerator(
            self.flood_model_path,
            self.hydromt_path,
            self.terrain_bounding_box,
            self.start_time,
            self.end_time,
            self.flood_type,
            self.polygons,
            self.vectors
        )

        # Generate parameter files
        output_dir = parameters_files_generator.parameter_files_generator()

        return output_dir

    # pylint: disable=useless-type-doc,useless-param-doc
    def flood_model_simulations_generator(self, _output_dir: Path | None) -> Path:
        """
        Generate flood simulations by running flood model

        Parameters
        ----------
        _output_dir : Path
            WARNING: THIS IS IGNORED, output directory is generated within function.
            Awaiting reconciling how directories are handled in the application

            _The path to the output directory, to allow for serving.

        Returns
        -------
        Path
            The Flood Model maximum extents raster file
        """
        # Set up path to log file
        log_file_path = self.flood_model_path / "simulation_log.log"

        # Get the output folder path
        output_folder_path = self.flood_model_path / "output"

        # Copy flood model folder, including executable and .dlls
        shutil.copytree(EnvVariable.FLOOD_MODEL_DIR, output_folder_path, dirs_exist_ok=True)

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

        # BG flood command
        output_executable = output_folder_path / flood_model_exe_path.name
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
        output_nc = output_folder_path / "output.nc"
        output_tif = convert_nc_to_gtiff(output_nc)

        return output_tif

    def flood_model_executor(self) -> int:
        """
        Generate necessary inputs for flood model

        Returns
        -------
        int
            The model output ID of the flood model run
        """
        # Get original path
        original_path = self.flood_model_path.parents[1] / "original_scenario/hydrodynamic_process"

        # Four cases:
        # 1. Original scenario (polygon=None, vector=None)
        # 2. Polygon=None, vector!=None
        # 3. Polygon!=None, vector=None
        # 4. Polygon!=None, vector!=None
        # This 'if' includes 1
        if self.polygons is None and self.vectors is None:
            # Generate injection points for flood model
            self.injection_points_for_flood_model_generator()

            # Generate precipitation data for flood model
            if self.flood_type == FloodType.PLUVIAL:
                self.precipitation_data_for_flood_model_generator()

            # Generate parameter files for flood model
            self.parameter_files_for_flood_model_generator()

        # This 'elif' includes 3 and 4
        elif self.polygons is not None or self.vectors is None:
            # Generate injection points for flood model
            self.injection_points_for_flood_model_generator()

            # Copy precipitation
            if self.flood_type == FloodType.PLUVIAL:
                shutil.copy2(
                    original_path / "precipitation_dynamic.nc",
                    self.flood_model_path / "precipitation_dynamic.nc"
                )

            # Generate parameter files for flood model
            self.parameter_files_for_flood_model_generator()

        # This 'else' includes 2
        else:
            # Copy injection points
            shutil.copy2(
                original_path / "injection_points_flow.csv",
                self.flood_model_path / "injection_points_flow.csv"
            )
            for original_file in original_path.glob("injection_points.*"):
                shutil.copy2(
                    original_file,
                    self.flood_model_path / original_file.name
                )

            # Copy precipitation
            if self.flood_type == FloodType.PLUVIAL:
                shutil.copy2(
                    original_path / "precipitation_dynamic.nc",
                    self.flood_model_path / "precipitation_dynamic.nc"
                )

            # Generate parameter files for flood model
            self.parameter_files_for_flood_model_generator()

        # Generate parameter files for flood model
        output_dir = self.parameter_files_for_flood_model_generator()

        # Generate simulations by running flood model
        max_gtiff = self.flood_model_simulations_generator(output_dir)

        # Add the results to the database and geoserver
        model_output_id = self.serve_flood_model_outputs(max_gtiff)
        self.serve_injection_points(model_output_id)
        return model_output_id
