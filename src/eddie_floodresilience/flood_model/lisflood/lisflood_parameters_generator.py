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

"""This script sets parameter files for LISFLOOD-FP model runs."""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np

from eddie.digitaltwin.utils import LogLevel, setup_logging
from ..flood_model_parameters_generator import FloodModelParametersGenerator, FloodType

setup_logging(LogLevel.DEBUG)
log = logging.getLogger(__name__)


class LisfloodParametersGenerator(FloodModelParametersGenerator):
    """This class is to generate parameter files for flood model"""

    def bci_generator(self) -> None:
        """Generate bci files - where the locations of injection points are defined"""
        bci_path = self.flood_model_path / "bci.bci"
        log.info(f"Genarating bci file {bci_path}")

        # Read and add crs injection point shapefiles
        injection_points = gpd.read_file(self.flood_model_path / "injection_points.shp")
        injection_points = injection_points.to_crs(2193)

        # Write out files
        with open(bci_path, "w", encoding="utf-8") as bci_parameter:
            # Set up boundaries before creating the bci files
            xmin = self.terrain_bounding_box.bounds[0]
            ymin = self.terrain_bounding_box.bounds[1]
            xmax = self.terrain_bounding_box.bounds[2]
            ymax = self.terrain_bounding_box.bounds[3]

            # Add coordinates to boundary directions
            boundary_edges = [
                ['N', xmin, xmax, 'FREE', ''],
                ['S', xmin, xmax, 'FREE', ''],
                ['E', ymin, ymax, 'FREE', ''],
                ['W', ymin, ymax, 'FREE', '']
            ]

            # Write into bci file
            for edge_coordinates in boundary_edges:
                # pylint: disable=consider-using-f-string
                edge_text = '{0[0]:<5}{0[1]:<20}{0[2]:<20}{0[3]:<7}{0[4]:<5}\n'.format(edge_coordinates)
                bci_parameter.write(edge_text)

            # Get injection points ID from injection_points_flow data
            buffer_distance = 12
            tolerance = 5

            # Get injection points ID
            injection_points_id = [
                int(col.split('_')[1]) for col in self.injection_points_flow.columns if col.startswith('Q_')
            ]

            # Write out injection points coordinates
            for each_id in injection_points_id:
                injection_points_id = injection_points[injection_points['FID'] == f'Q_{each_id}'].iloc[0]

                # Get xy coordinates and aoi coordinates
                xy_coords = [injection_points_id.geometry.x, injection_points_id.geometry.y]
                aoi_coords = [xmin, ymin, xmax, ymax]

                # Move points inside area of interest
                x, y = self.move_points_inside_aoi(aoi_coords, xy_coords, buffer_distance, tolerance)

                # Set up coordinates text for points
                injection_points_text = f'{"P":<5}{x:<20}{y:<20}{"QVAR":<7}{f"Q_{each_id}":<5}\n'

                # Write inside bci file
                bci_parameter.write(injection_points_text)

            if self.tide_df is not None:
                # Add tide geometry
                # Read tidal point
                onshore_tidal_point_gdf = gpd.read_file(
                    self.flood_model_path / "tidal_point.shp"
                )
                # Setup tidal point format for bci
                onshore_tidal_point = onshore_tidal_point_gdf.geometry.iloc[0]

                # Design text for tidal point format
                onshore_tidal_point_text = (
                    f"{'P':<5}"
                    f"{onshore_tidal_point.x:<20.3f}"
                    f"{onshore_tidal_point.y:<20.3f}"
                    f"{'HVAR':<7}"
                    f"{'Tide':<5}\n"
                )
                # Write into bci
                bci_parameter.write(onshore_tidal_point_text)

    def bdy_generator(self) -> None:
        """Generate bdy files - where the flow data of injection points are stored"""
        # Path of flow data (bdy) for flood model
        bdy_name = self.flood_model_path / "bdy.bdy"

        log.info(f"Generating BDY file {bdy_name}")
        # Copy injection flow dataframe
        flow_df = self.injection_points_flow.copy(deep=True)

        # Create "time" and "seconds" column
        flow_df['time'] = pd.to_datetime(flow_df['time'])
        flow_df['seconds'] = (
            flow_df['time'] - flow_df['time'].iloc[0]
        ).dt.total_seconds().astype(int)

        # Identify only flow columns started with Q
        flow_columns = [
            col for col in flow_df.columns if col.startswith('Q_')
        ]

        # Adjust flow values
        # At the moment, we use 8m resolution
        # This will be adjusted in the future
        flow_df[flow_columns] = (flow_df[flow_columns] / 8).round(4)

        # Write out flow data for injection points
        with open(bdy_name, "w", encoding="utf-8") as discharge_tide:

            discharge_tide.write("LISFLOOD-FP setup\n")

            # Write flow data
            for flow_column in flow_columns:
                # Create new line for each flow value
                discharge_tide.write(flow_column + "\n")
                discharge_tide.write(f'{flow_df.shape[0]:<20}seconds\n')

                # Write flow values
                for i in range(flow_df.shape[0]):
                    value = flow_df.at[i, flow_column]
                    sec = flow_df.at[i, 'seconds']

                    line = f"{value:<20.4f}{sec:.0f}\n"
                    discharge_tide.write(line)

            if self.tide_df is not None:
                # Write tide data (only one point)
                discharge_tide.write("Tide\n")
                discharge_tide.write(f'{self.tide_df.shape[0]:<20}seconds\n')
                # Write tide values
                for _idx, row in self.tide_df.iterrows():
                    line = f"{row.value:<20.4f}{row.seconds:.0f}\n"
                    discharge_tide.write(line)

    def write_flood_model_parameter_file(self) -> Path:
        """
        Generate par files - where all the parameter data are navigated
        Returns
        -------
        Path
            Directory of output files as configured in parameter files
        """
        par_file_path = self.flood_model_path / "par.par"
        log.info(f"Generating par file {par_file_path}")
        # Create output directory
        output_directory = self.optional_output_generator()

        # Path to bdy file
        bdy = str(self.flood_model_path / "bdy.bdy")

        # Path to bci file
        bci = str(self.flood_model_path / "bci.bci")

        # Path to z file
        z = str(self.flood_model_path / "z.asc")

        # Path to Manning's n
        n = str(self.flood_model_path / "manning.asc")

        # Create parameters list
        parameters_list = [
            ('resroot', 'out'),
            ('dirroot', str(output_directory)),
            ('saveint', 21600),
            ('massint', 500),
            ('sim_time', f'{self.seconds}'),
            ('initial_tstep', 5),
            ('bcifile', bci),
            ('bdyfile', bdy),
            ('DEMFile', z),
            ('manningfile', n)
        ]
        # For pluvial
        if self.flood_type == FloodType.PLUVIAL:
            # Add pluvial parameter
            rainfile = str(self.flood_model_path / "precipitation_dynamic.nc")
            parameters_list.append(('dynamicrainfile', rainfile))

        # Write into array
        parameters_array = np.array(parameters_list)

        # Write PAR file
        with open(par_file_path, "w", encoding="utf-8") as parameters:
            for each_parameter in range(parameters_array.shape[0]):
                data_parameter = parameters_array[each_parameter]
                text_parameter = f'{data_parameter[0]:<20}{data_parameter[1]}\n'
                parameters.write(text_parameter)
            parameters.write('acceleration\ndrain_nodata\n\n')
        return output_directory

    def parameter_files_generator(self) -> Path:
        """
        Generate parameter files for flood model

        Returns
        -------
        Path
            Directory of output files as configured in parameter files
        """
        self.write_injection_point_files()

        # Generate par file
        output_dir = self.write_flood_model_parameter_file()
        return output_dir

    def write_injection_point_files(self) -> None:
        """Write injection point files for Lisflood."""
        # Generate bci file
        self.bci_generator()

        # Generate bdy file
        self.bdy_generator()
