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

"""Runs LISFLOOD-FP flood model"""

from pathlib import Path
import logging
import platform
import subprocess

from eddie.digitaltwin.utils import LogLevel, setup_logging

from src.eddie_floodresilience.config import EnvVariable
from .lisflood_inputs_generator import TerrainFloodModelGenerator
from .lisflood_parameters_generator import LisfloodParametersGenerator
from .lisflood_precipitation import LisfloodPrecipitationGenerator, LisfloodPrecipitationFloodModelGenerator
from ..flood_model_siumulations_generator import BaseFloodModelSimulationsGenerator

setup_logging(LogLevel.DEBUG)
log = logging.getLogger(__name__)


class LisFloodModelSimulationsGenerator(BaseFloodModelSimulationsGenerator):
    """This class is to generate flood model simulations."""  # pylint: disable=too-many-instance-attributes

    def terrain_data_for_flood_model_generator(self) -> None:
        """Generate terrain data for flood model (LISFLOOD-FP)"""
        # Call out class used to generate terrain data for flood model
        terrain_data_for_flood_model = TerrainFloodModelGenerator(
            self.flood_model_path,
            self.hydromt_path,
            self.river_name,
            self.terrain_crs_clipped,
            self.adjust_manning,
            self.crs
        )

        # Generate terrain data for flood model
        terrain_data_for_flood_model.execute_terrain_data_generator()

    def precipitation_data_for_flood_model_generator(self) -> None:
        """Generate precipitation data for flood model"""
        # Call out class used to generate precipitation data
        precipitation_generator = LisfloodPrecipitationGenerator(
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
        precipitation_data_for_flood_model = LisfloodPrecipitationFloodModelGenerator(
            self.flood_model_path,
            precipitation_data
        )

        # Generate precipitation data for flood model
        precipitation_data_for_flood_model.precipitation_for_flood_model_generator()

    def parameter_files_for_flood_model_generator(self) -> Path:
        """
        Generate parameters files for flood model

        Return
        ------
        Path
            The path to the output directory generated for these parameters
        """
        # Call out class used to generate parameter files
        parameters_files_generator = LisfloodParametersGenerator(
            self.flood_model_path,
            self.terrain_bounding_box,
            self.start_time,
            self.end_time,
            self.polygons,
            self.vectors
        )

        # Generate parameter files
        output_dir = parameters_files_generator.parameter_files_generator()
        return output_dir

    def flood_model_simulations_generator(self, output_dir: Path) -> int:
        """
        Generate flood simulations by running flood model

        Parameters
        ----------
        output_dir : Path
            The path to the output directory, to allow for serving.

        Returns
        -------
        int
            The Flood Model output ID
        """
        # Set up path to log file
        log_file_path = self.flood_model_path / "simulation_log.log"

        # Set up path to parameters' file
        par_file_path = str(self.flood_model_path / "par.par")

        # Identify the LISFLOOD-FP executable, accounting for OS differences
        operating_system = platform.system()
        linux_path = EnvVariable.HYDROMT_PATH / "lisflood"
        match operating_system:
            case "Windows":
                lisflood_path = EnvVariable.HYDROMT_PATH / "lisflood_v8_1_0.exe"
            case "Linux":
                lisflood_path = linux_path
            case _:
                lisflood_path = linux_path
                log.warning(
                    f"{operating_system} is not officially supported. Only Windows and Linux are officially supported.")
                log.warning(f"Attempting to run LISFLOOD-FP linux script in {operating_system}")

        # Flood simulation command
        flood_simulation_command = [
            lisflood_path,
            "-v",
            par_file_path
        ]

        # Generate flood model simulations
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            log.info("Running LISFLOOD-FP flood simulation")
            subprocess.run(
                flood_simulation_command,
                stdout=log_file,
                stderr=subprocess.STDOUT,  # add error into log file if appears
                check=True
            )

        model_output_id = self.serve_flood_model_outputs(output_dir)
        return model_output_id

    def flood_model_executor(self) -> int:
        """
        Generate necessary inputs for flood model

        Returns
        -------
        int
            The Flood Model output ID
        """
        # Four cases:
        # 1. Original scenario (polygon=None, vector=None)
        # 2. Polygon=None, vector!=None
        # 3. Polygon!=None, vector=None
        # 4. Polygon!=None, vector!=None
        # This 'if' includes 1
        if self.polygons is None and self.vectors is None:
            # Generate terrain data for flood model
            self.terrain_data_for_flood_model_generator()

            # Generate injection points for flood model
            self.injection_points_for_flood_model_generator()

            # Generate precipitation data for flood model
            # self.precipitation_data_for_flood_model_generator()

            # Generate parameter files for flood model
            output_dir = self.parameter_files_for_flood_model_generator()

            # Generate simulations by running flood model
            model_output_id = self.flood_model_simulations_generator(output_dir)

        # This 'elif' includes 3 and 4
        elif self.polygons is not None or self.vectors is None:
            # Generate injection points for flood model
            self.injection_points_for_flood_model_generator()

            # Generate parameter files for flood model
            output_dir = self.parameter_files_for_flood_model_generator()

            # Generate simulations by running flood model
            model_output_id = self.flood_model_simulations_generator(output_dir)

        # This 'else' includes 2
        else:
            # Generate terrain data for flood model
            self.terrain_data_for_flood_model_generator()

            # Generate parameter files for flood model
            output_dir = self.parameter_files_for_flood_model_generator()

            # Generate simulations by running flood model
            model_output_id = self.flood_model_simulations_generator(output_dir)
        return model_output_id
