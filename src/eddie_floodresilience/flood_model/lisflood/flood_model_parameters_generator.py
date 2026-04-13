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
        start_time: str,
        end_time: str
    ) -> None:
        """
        Generate parameter files for flood model

        Parameters
        ----------
        flood_model_path : Path
            Directory to folder storing flood model data
        terrain_bounding_box : Polygon
            Bounding's box of terrain data
        start_time : str
            String of starting time details. Format is "yyyy-mm-ddThh:mm:ss"
        end_time : str
            String of ending time details.
        """
        self.flood_model_path = flood_model_path
        self.terrain_bounding_box = terrain_bounding_box
        self.injection_points_flow = pd.read_csv(fr"{self.flood_model_path}\injection_points_flow.csv")
        self.start_time = start_time
        self.end_time = end_time
        
    def bci_generator(self) -> None:
        """Generate bci files - where the locations of injection points are defined"""
        
        # At the moment there is only flow data
        # The tide data will be added in the future
        
        # Read and add crs injection point shapefiles
        injection_points = gpd.read_file(fr"{self.flood_model_path}\injection_points.shp")
        injection_points = injection_points.to_crs(2193)
        
        # Write out files
        with open(fr"{self.flood_model_path}\bci.bci", "w") as bci_parameter:
            # Set up boundaries before creating the bci files
            # It is set to Northern direction at the moment
            # It will be changed into automatic step
            edge_coordinates = [
                'N', 
                self.terrain_bounding_box.bounds[0],
                self.terrain_bounding_box.bounds[2],
                'FREE', ''
            ]
            edge_text = '{0[0]:<5}{0[1]:<20}{0[2]:<20}{0[3]:<7}{0[4]:<5}\n'.format(edge_coordinates)
            bci_parameter.write(edge_text)
            
            # Get injection points ID from injection_points_flow data
            injection_points_id = [
                int(col.split('_')[1]) for col in self.injection_points_flow.columns if col.startswith('Q_')
            ]
            
            # Write out injection points coordinates
            for each_id in injection_points_id:
                injection_points_id = injection_points[injection_points['FID'] == f'Q_{each_id}'].iloc[0]
                injection_points_boundary = [
                    'P', 
                    injection_points_id.geometry.x, 
                    injection_points_id.geometry.y,
                    'QVAR', f"Q_{each_id}"
                ]
                injection_points_text = '{0[0]:<5}{0[1]:<20}{0[2]:<20}{0[3]:<7}{0[4]:<5}\n'.format(
                    injection_points_boundary
                )
                bci_parameter.write(injection_points_text)
                
                
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
        
        # Write out flow data for injection points
        with open(fr"{self.flood_model_path}\bdy.bdy", "w") as discharge_tide:

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
        # Format simulated time
        formatted_start_time = datetime.strptime(self.start_time, "%Y-%m-%dT%H:%M:%S")
        formatted_end_time = datetime.strptime(self.end_time, "%Y-%m-%dT%H:%M:%S")
        
        # Compute simulated time in seconds
        seconds = int((formatted_end_time - formatted_start_time).total_seconds())
        
        return seconds
        
    def par_generator(self) -> None:
        """Generate par files - where all the parameter data are navigated"""
        
        # Path to output
        output = fr"{self.flood_model_path}\output"
        Path(output).mkdir(parents=True, exist_ok=True)
        
        # Path to bdy file
        bdy = fr"{self.flood_model_path}\bdy.bdy"
        
        # Path to bci file
        bci = fr"{self.flood_model_path}\bci.bci"
        
        # Path to DEM
        z = fr"{self.flood_model_path}\z.asc"
        
        # Path to Manning's n
        n = fr"{self.flood_model_path}\manning.asc"
        
        # Path to precipitation
        precipitation = fr"{self.flood_model_path}\precipitation_dynamic.nc"
        
        # Simulated time in seconds
        seconds = self.simulated_seconds_generator()
        
        # Create parameters list
        parameters_list = [
            ('resroot', 'out'),
            ('dirroot', output),
            ('saveint', 3600),
            ('massint', 100),
            ('sim_time', f'{seconds}'),
            ('initial_tstep', 2),
            ('bcifile', bci),
            ('bdyfile', bdy),
            ('DEMFile', z),
            ('manningfile', n),
            ('dynamicrainfile', precipitation)
        ]
        
        # Write into array
        parameters_array = np.array(parameters_list)

        # Write PAR file
        with open(fr"{self.flood_model_path}\par.par", "w") as parameters:
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
        
                
            
        