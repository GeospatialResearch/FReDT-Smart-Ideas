# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:23:13 2026

@author: mng42
"""

from pathlib import Path
import sys


from .terrain_data_manipulator import TerrainFilter
from .terrain_attributes_generator import (StreamTopologyGenerator,
                                           StreamHydraulicsGenerator)
from .terrain_data_for_wflow_preparator import TerrainDataWflowPreparator

class TerrainDataWflowGenerator:
    """This class is to generate terrain data for wflow model"""
    
    def __init__(
            self,
            terrain_path: Path,
            hydromt_path: Path,
            river_name: str,
            subbasin: list,
            resolution: float = 100,
            threshold: int = 1000
        ) -> None:
        """
        Generate terrain data for wflow model
        
        Parameters
        ----------
        terrain_path : Path
            A directory to where the terrain data are stored
        hydromt_path : Path
            A directory to where all necessary files are stored to run wflow model
        river_name : str
            Name of directory to where the river information files are stored
        subbasin : list
            Outlet coordinates
        resolution: float = 100
            Resolution to resample data. Default is 100m in crs 2193)
        threshold: int = 1000
            Minimum number of cells/up-slope area required to initiate and main a channel.
            Default is 1000
        """
        self.terrain_path = terrain_path
        self.hydromt_path = hydromt_path
        self.river_name = river_name
        self.subbasin = subbasin
        self.resolution = resolution
        self.threshold = threshold
        
    def filtered_terrain_generator(self) -> None:
        """Filter terrain data to generate necessary data for wflow model"""
        filtered_terrain = TerrainFilter(
            self.terrain_path,
            self.hydromt_path,
            self.river_name
        )
        filtered_terrain.filter_dem_for_wflow()
        
    def stream_topology_generator(self) -> None:
        """Generate stream topology for wflow model"""
        stream_topology = StreamTopologyGenerator(
            self.terrain_path,
            self.resolution,
            self.threshold
        )
        stream_topology.dataframe_upstream_area_strahler_geometry_generator()
        
    def stream_hydraulic_generator(self) -> None:
        """Generate stream hydraulic for wflow model"""
        stream_hydraulic = StreamHydraulicsGenerator(
            path=self.terrain_path,
            hydromt_path=self.hydromt_path,
            river_name=self.river_name,
            subbasin=self.subbasin,
            streams_bankfull_stage=1.5,
            resolution=self.resolution,
            threshold=self.threshold
        )
        stream_hydraulic.dataframe_stream_bankfull_width_discharge_generator()
        
    def terrain_for_wflow_preparator(self) -> None:
        """Prepare terrain data for wflow model"""
        terrain_for_wflow = TerrainDataWflowPreparator(self.terrain_path)
        terrain_for_wflow.terrain_data_for_wflow_preparator()
        
    def terrain_for_wflow_generator(self) -> None:
        """Generate terrain data for wflow model"""
        # Fitler terrain data to generate necessary data for wflow model
        self.filtered_terrain_generator()
        
        # Generate stream topology for wflow model
        self.stream_topology_generator()
        
        # Generate stream hydraulic for wflow model
        self.stream_hydraulic_generator()
        
        # Generate and prepare terrain data for wflow model
        self.terrain_for_wflow_preparator()

