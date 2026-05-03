# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:23:13 2026

@author: mng42
"""

from pathlib import Path
import sys
sys.path.append(r"C:\Users\mng42\spyder_hydromt\digital_twin_codes\for_both")

from src.eddie_floodresilience.hydrological_and_hydrodynamic_pipeline import HydraulicAndHydrodynamicPipeline


hydro_combination_path = Path(r"D:\Digital_Twin_data\hydraulic_hydrodynamic_path_001")
outlet_gauge_locations_filename = 'river_points_with_outlets_002'
precipitation_path = Path(r"H:\Barra/Whirinaki\rainfall_gauges_HIRDS_004")
start_time = "1998-01-01T00:00:00"
end_time = "1999-03-01T00:00:00"

subbasin = [173.46365,-35.45662]
bbox = [173.44134,-35.61760, 173.77608,-35.11105]
num_threads = 6
flood_aoi_boundary = [1641145.361, 6072406.885, 1642792.613, 6076268]

strord = 4
resolution = 0.00045
threshold = 1000
width_rate_control = 1/20
discharge_rate_control = 1
crs = 2193

# Set up hydraulic and hydrodynamic pipeline
hydraulic_hydrodynamic_pipeline = HydraulicAndHydrodynamicPipeline(
    hydro_combination_path,
    outlet_gauge_locations_filename,
    precipitation_path,
    start_time,
    end_time,

    subbasin,
    bbox,
    num_threads,
    flood_aoi_boundary,

    strord,
    resolution,
    threshold,
    width_rate_control,
    discharge_rate_control,
    crs
)

# Generate hydraulic and hydrodynamic simulations
hydraulic_hydrodynamic_pipeline.hydraulic_and_hydrodynamic_simulation_generator()
