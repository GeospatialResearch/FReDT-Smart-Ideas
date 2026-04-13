# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:11:33 2026

@author: mng42
"""

from pathlib import Path
import sys
sys.path.append(r"C:\Users\mng42\spyder_hydromt\digital_twin_codes\for_both")

from terrain_data_for_wflow_generator import TerrainDataWflowGenerator

from wflow_simulations_generator import WflowSimulationsGenerator
from flood_model_simulations_generator import FloodModelSimulationsGenerator


# TERRAIN ---------------------------------------------------------------------

terrain_path = Path(r"D:\Digital_Twin_data\terrain_data_whirinaki_001")
outlet_gauge_locations_filename = 'river_points_with_outlets_002'
terrain_data_for_wflow = TerrainDataWflowGenerator(
    terrain_path,
    outlet_gauge_locations_filename,
    0.00045,
    1000,
    1/20, 1
)
terrain_data_for_wflow.terrain_for_wflow_generator()

# WFLOW -----------------------------------------------------------------------
hydromt_path = Path(r"D:\Digital_Twin_data\necessary_data")
wflow_model_path = Path(r"D:\Digital_Twin_data\terrain_data_whirinaki_001")
precipitation_path = Path(r"H:\Barra/Whirinaki\merge_gauges_HIRDS_004")
start_time = "1998-01-01T00:00:00"
end_time = "1999-03-01T00:00:00"

subbasin = [173.46365,-35.45662]
strord = 4
bbox = [173.44134,-35.61760, 173.77608,-35.11105]

num_threads = 6
resolution = 0.00045

# Set up wflow model generator
wflow_simulations_generator = WflowSimulationsGenerator(
    hydromt_path,
    wflow_model_path,
    precipitation_path,
    start_time,
    end_time,
    subbasin,
    strord,
    bbox,
    num_threads,
    resolution
)

# Generate wflow model simulation
wflow_simulations_generator.wflow_model_simulations_pipeline()


# LISFLOOD-FP -----------------------------------------------------------------

flood_model_path = Path(r"D:\Digital_Twin_data\terrain_data_whirinaki_001")
catchment_model_path = Path(r"D:\Digital_Twin_data\terrain_data_whirinaki_001")
precipitation_path = Path(r"H:\Barra\Whirinaki\rainfall_gauges_HIRDS_004")
aoi_boundary = [1641145.361, 6072406.885, 1642792.613, 6076268]
start_time = "1999-01-20T00:00:00"
end_time = "1999-01-22T12:00:00"
crs = 2193

flood_model_pipeline = FloodModelSimulationsGenerator(
    flood_model_path=flood_model_path,
    catchment_model_path=catchment_model_path,
    precipitation_path=precipitation_path,
    aoi_boundary=aoi_boundary,
    start_time = start_time,
    end_time = end_time,
    crs = crs 
)

# Generate inputs for flood model
flood_model_pipeline.flood_model_executor()