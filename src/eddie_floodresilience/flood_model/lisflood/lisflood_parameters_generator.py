# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:16:47 2026

@author: mng42
"""

from osgeo import gdal # Import gdal before rasterio

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

import pandas as pd

from datetime import datetime

from pathlib import Path

class ParametersFloodModelGenerator():
    """This class is to generate parameter files for flood model"""
    
    def __init__(
        self,
        flood_model_path: Path,
        terrain_bounding_box: Polygon,
        start_time: datetime,
        end_time: datetime,
        polygons: str = None,
        vectors: str = None
    ) -> None:
        """
        Generate parameter files for flood model

        Parameters
        ----------
        flood_model_path : Path
            Directory to folder storing flood model data
        terrain_bounding_box : Polygon
            Bounding's box of terrain data
        start_time : datetime
            Starting time details. Format is "yyyy-mm-ddThh:mm:ss"
        end_time : datetime
            Ending time details.
        polygons : str = None
            Name of polygon file that is used to change the landcover information.
            This polygon dataframe has 'landcover' column with new values
        vectors : str = None
            Name of vector file that is used to change the elevation information.
            This vector dataframe has 'value' column to specify increasing or decreasing elevation,
            and 'distance' column to specify how smooth to decrease elevation.
        """
        self.flood_model_path = flood_model_path
        self.terrain_bounding_box = terrain_bounding_box
        self.injection_points_flow = pd.read_csv(self.flood_model_path / "injection_points_flow.csv")
        self.start_time = start_time
        self.end_time = end_time
        self.polygons = polygons
        self.vectors = vectors

    def move_points_inside_aoi(
            self,
            aoi_coords,
            xy_coords,
            buffer_distance,
            tolerance
    ):
        """
        Move points inside aoi

        Parameters
        ----------
        aoi_coords : list
            Coordinates of area of interest
        xy_coords : list
            Coordinates of x and y. X and y of each injection point
        buffer_distance : float
            Amount that the points are moved inside
        tolerance : float
            How much differences in distance between injection points and coordinates of area of interest

        Returns
        -------
        xy_coords : list
            New coordinates of x and y. X and y of each injection point
        """
        # Extract each x and y
        x, y = xy_coords

        # Extract xmin, ymin, xmax, ymax
        xmin, ymin, xmax, ymax = aoi_coords

        # Move points inside DEM
        if abs(y - ymax) <= tolerance:
            y -= buffer_distance

        elif abs(y - ymin) <= tolerance:
            y += buffer_distance

        elif abs(x - xmax) <= tolerance:
            x -= buffer_distance

        elif abs(x - xmin) <= tolerance:
            x += buffer_distance

        return x, y

        
    def bci_generator(self) -> None:
        """Generate bci files - where the locations of injection points are defined"""
        
        # At the moment there is only flow data
        # The tide data will be added in the future
        
        # Read and add crs injection point shapefiles
        injection_points = gpd.read_file(self.flood_model_path / "injection_points.shp")
        injection_points = injection_points.to_crs(2193)
        
        # Write out files
        with open(self.flood_model_path / "bci.bci", "w") as bci_parameter:
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
                edge_text = ('{0[0]:<5}{0[1]:<20}{0[2]:<20}{0[3]:<7}{0[4]:<5}\n').format(edge_coordinates)
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
                injection_points_boundary = [
                    'P', 
                    x,
                    y,
                    'QVAR', f"Q_{each_id}"
                ]
                injection_points_text = '{0[0]:<5}{0[1]:<20}{0[2]:<20}{0[3]:<7}{0[4]:<5}\n'.format(
                    injection_points_boundary
                )

                # Write inside bci file
                bci_parameter.write(injection_points_text)

    def file_increment_generator(
            self,
            filename: str
    ):
        """
        Generate increasing files

        Parameters
        ----------
        filename : str
            Filename accords to scenario and order of that scenario

        Returns
        -------
        file_directory : Path
            A directory to the file which has just been named
        """

        number_ids = [
            int(f.stem.split("_")[-1])
            for f in Path(self.flood_model_path).glob(f"{filename}_*")
        ]

        # Get next output
        file_number = max(number_ids, default=0) + 1

        # Output path
        file_directory = Path(self.flood_model_path) / f"{filename}_{file_number:03d}"

        return file_directory

    def bdy_generator(self) -> None:
        """Generate bdy files - where the flow data of injection points are stored"""
        
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

        # Path of flow data (bdy) for flood model
        if self.polygons is not None:
            bdy_name = str(
                self.file_increment_generator(f"bdy_landcover")
            ) + ".bdy"
        else:
            bdy_name = str(self.flood_model_path / "bdy.bdy")

        # Write out flow data for injection points
        with open(bdy_name, "w") as discharge_tide:

            discharge_tide.write("LISFLOOD-FP setup\n")

            for flow_column in flow_columns:

                discharge_tide.write(flow_column + "\n")
                discharge_tide.write(
                    '{0:<20}seconds\n'.format(flow_df.shape[0])
                )

                for i in range(flow_df.shape[0]):
                    value = flow_df.at[i, flow_column]
                    sec = flow_df.at[i, 'seconds']

                    line = f"{value:<20}{sec:.0f}\n"
                    discharge_tide.write(line)
                    
    def simulated_seconds_generator(self):
        """
        Generate simulated time in seconds
        
        Returns
        -------
        seconds : int
            Simulated time in seconds from starting to ending times
        
        """
        # Compute simulated time in seconds
        seconds = int((self.end_time - self.start_time).total_seconds())
        
        return seconds

    def optional_output_generator(self) -> str:
        """
        Set up options for outputs according to sceanrios
        and create output directory for flood modelling outputs

        Returns
        -------
        output_directory : str
            String of directory of flood model outputs
        """
        # Both landcover and elevation solutions
        if self.polygons is not None and self.vectors is not None:
            output = self.file_increment_generator("output_landcover_elevation")

        # Only landcover solution
        elif self.polygons is not None:
            output = self.file_increment_generator("output_landcover")

        # Only elevation solution
        elif self.vectors is not None:
            output = self.file_increment_generator("output_elevation")

        # Original scenario
        else:
            output = self.flood_model_path / "output"

        # Create output (if not available)
        output.mkdir(parents=True, exist_ok=True)

        # Get output directory
        output_directory = str(output)

        return output_directory
        
    def par_generator(self) -> None:
        """Generate par files - where all the parameter data are navigated"""

        # Create output directory
        output_directory = self.optional_output_generator()
        
        # Path to bdy file
        if self.polygons is not None:
            bdy = str(
                max(
                Path(self.flood_model_path).glob("bdy_landcover_*.bdy"),
                default=Path(self.flood_model_path) / "bdy_landcover_001.bdy"
                )
            )
        else:
            bdy = str(self.flood_model_path / "bdy.bdy")

        # Path to bci file
        bci = str(self.flood_model_path / "bci.bci")

        if self.vectors is not None:
            # Path to DEM
            z = str(
                max(
                Path(self.flood_model_path).glob("z_elevation_*.asc"),
                default=Path(self.flood_model_path) / "z_elevation_001.asc"
                )
            )
        else:
            # Path to DEM
            z = str(self.flood_model_path / "z.asc")

        # Path to Manning's n
        n = str(self.flood_model_path / "manning.asc")
        
        # Path to precipitation
        precipitation = str(self.flood_model_path / "precipitation_dynamic.nc")
        
        # Simulated time in seconds
        seconds = self.simulated_seconds_generator()
        
        # Create parameters list
        parameters_list = [
            ('resroot', 'out'),
            ('dirroot', output_directory),
            ('saveint', 21600),
            ('massint', 500),
            ('sim_time', f'{seconds}'),
            ('initial_tstep', 5),
            ('bcifile', bci),
            ('bdyfile', bdy),
            ('DEMFile', z),
            ('manningfile', n)
            # ('dynamicrainfile', precipitation)
        ]
        
        # Write into array
        parameters_array = np.array(parameters_list)

        # Write PAR file
        with open(self.flood_model_path / "par.par", "w") as parameters:
            for each_parameter in range(parameters_array.shape[0]):
                data_parameter = parameters_array[each_parameter]
                text_parameter = '{0[0]:<20}{0[1]}\n'.format(data_parameter)
                parameters.write(text_parameter)
            parameters.write('acceleration\ndrain_nodata\n\n')
            

    def parameters_files_generator(self) -> None:
        """Generate parameter files for flood model"""
        
        # Generate bci file
        self.bci_generator()
        
        # Generate bdy file
        self.bdy_generator()
        
        # Generate par file
        self.par_generator()

        