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
from pathlib import Path

from eddie.digitaltwin.utils import LogLevel, setup_logging

from src.eddie_floodresilience.config import EnvVariable
from src.eddie_floodresilience.flood_model.serve_model import add_model_output_to_geoserver, convert_nc_to_gtiff
from .bgflood_parameters_generator import BGFloodParametersGenerator
from .bgflood_precipitation import BGFloodPrecipitationGenerator, BGFloodPrecipitationFloodModelGenerator
from ..flood_model_siumulations_generator import BaseFloodModelSimulationsGenerator

setup_logging(LogLevel.DEBUG)
log = logging.getLogger(__name__)


class BGFloodModelSimulationsGenerator(BaseFloodModelSimulationsGenerator):
    """This class is to generate flood model simulations"""  # pylint: disable=too-many-instance-attributes

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

    def parameter_files_for_flood_model_generator(self) -> None:
        """Generate parameters files for flood model"""
        # Call out class used to generate parameter files
        parameters_files_generator = BGFloodParametersGenerator(
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

    def flood_model_simulations_generator(self, _output_dir: Path | None) -> int:
        """
        Generate flood simulations by running flood model

        Parameters
        ----------
        _output_dir : Path
            The path to the output directory, to allow for serving.

        Returns
        -------
        int
            The Flood Model output ID
        """
        # Set up path to log file
        log_file_path = self.flood_model_path / "simulation_log.log"

        # Get the output folder path
        output_folder_path = self.find_output_folder_path()

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
        model_output_id = self.serve_flood_model_outputs(output_tif)
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
            self.parameter_files_for_flood_model_generator()

            # Generate precipitation data for flood model
            self.precipitation_data_for_flood_model_generator()

            # Generate simulations by running flood model
            model_output_id = self.flood_model_simulations_generator(None)

        # This 'elif' includes 3 and 4
        elif self.polygons is not None or self.vectors is None:
            # Generate injection points for flood model
            self.injection_points_for_flood_model_generator()

            # Generate parameter files for flood model
            self.parameter_files_for_flood_model_generator()

            # Generate simulations by running flood model
            model_output_id = self.flood_model_simulations_generator(None)

        # This 'else' includes 2
        else:
            # Generate parameter files for flood model
            self.parameter_files_for_flood_model_generator()

            # Generate simulations by running flood model
            model_output_id = self.flood_model_simulations_generator(None)
        return model_output_id
