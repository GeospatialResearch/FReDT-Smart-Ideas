# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:11:15 2026

@author: mng42
"""

from pathlib import Path
from datetime import datetime

from src.eddie_floodresilience.solutions.total_solutions import LandCoverSolution, ElevationSolution

from src.eddie_floodresilience.preprocessing.terrain_data_for_wflow_generator import TerrainDataWflowGenerator

from src.eddie_floodresilience.hydrological.wflow_simulations_generator import WflowSimulationsGenerator
from src.eddie_floodresilience.flood_model.lisflood.flood_model_simulations_generator import FloodModelSimulationsGenerator



class HydrologicalAndHydrodynamicPipeline:
    """This class is to generate hydrological and hydrodynamic results"""
    
    def __init__(
            self,
            hydro_combination_path: Path,
            outlet_gauge_locations_filename: str,

            forcing_path: Path,
            precipitation_path: Path,
            start_time: datetime,
            end_time: datetime,
            
            subbasin: list,
            bbox: list,
            num_threads: int,
            flood_aoi_boundary: list,

            polygons: str = None,
            vectors: str = None,
            strord: int = 4,
            resolution: float = 0.00045,
            threshold: int = 1000,
            width_rate_control: float = 2,
            discharge_rate_control: float = 1,
            crs: int = 2193
        ) -> None:
        """
        Generate hydrological and hydrodynamic results
        
        Parameters
        ----------
        hydro_combination_path : Path
            Directory to folder storing all necessary data
        outlet_gauge_locations_filename: str
            Filename of outlet gauge locations
        forcing_path: Path
            A directory to where the forcing files are stored
        precipitation_path: Path
            A directory to where the precipitation files are stored
        start_time : str
            Starting time of simulation.
            This should include the spin-up time. 
            Normally, it is 1-year before the flood event.
        end_time : str
            Ending time of simulation
            This should include some periods of time after the flood event.
            Normally, it is about 2-3 months.
        subbasin : list
            Outlet coordinates
        bbox : list
            Given bounding box coordinates that contains the sub-basin coordinates
        num_threads : int
            Number of threads that controls how fast the wflow model can run
        flood_aoi_boundary : list
            Boundaries' coordinates of area of interest.
            Format is [xmin, ymin, xmax, ymax]

        polygons : str = None
            Name of polygon file that is used to change the landcover information.
            This polygon dataframe has 'landcover' column with new values
        vectors : str = None
            Name of vector file that is used to change the elevation information.
            This vector dataframe has 'value' column to specify increasing or decreasing elevation,
            and 'distance' column to specify how smooth to decrease elevation.
        strord : int
            Minimum stream order
        resolution : float
            Resolution for flow data. 
            Default is 0.00045 (in crs 4326) ~ 50 m (in crs 2193)
        threshold: int = 1000
            Minimum number of cells/up-slope area required to initiate and main a channel.
            Default is 1000
        width_rate_control: float = 2
            The rate to control river width. Default is 2
        discharge_rate_control: float = 1
            The rate to control river discharge. Default is 1
        crs : int = 2193
            Targeted crs. The default is 2193 for NZTM.
        """
        # Set up necessary parameters
        self.hydro_combination_path = hydro_combination_path
        self.outlet_gauge_locations_filename = outlet_gauge_locations_filename
        self.forcing_path = forcing_path
        self.precipitation_path = precipitation_path
        self.start_time = start_time
        self.end_time = end_time
        self.subbasin = subbasin
        self.bbox = bbox
        self.num_threads = num_threads
        self.flood_aoi_boundary = flood_aoi_boundary

        self.polygons = polygons
        self.vectors = vectors
        self.strord = strord
        self.resolution = resolution
        self.threshold = threshold
        self.width_rate_control = width_rate_control
        self.discharge_rate_control = discharge_rate_control
        self.crs = crs
        
        self.hydromt_path = Path(r"D:\Digital_Twin_data\necessary_data")

    def total_solutions(self):
        """Develop solutions for flood risk resilience"""
        if self.polygons is not None:
            # Land cover/natural solution
            landcover_solution = LandCoverSolution(
                self.hydro_combination_path,
                self.hydromt_path,
                self.polygons
            )
            landcover_solution.apply_landcover_solution()

        if self.vectors is not None:
            # Elevation solution
            elevation_solution = ElevationSolution(
                self.hydro_combination_path,
                self.vectors
            )
            elevation_solution.apply_elevation_solution()

    def terrain_data_pipeline(self):
        """Generate terrain data for wflow and flood models"""
        # Set up terrain data generation system
        terrain_data = TerrainDataWflowGenerator(
            self.hydro_combination_path,
            self.outlet_gauge_locations_filename,
            self.resolution,
            self.threshold,
            self.width_rate_control,
            self.discharge_rate_control
        )
        
        # Generate terrain data
        terrain_data.terrain_for_wflow_generator()
        
    def wflow_data_pipeline(self):
        """Generate wflow model data for flood model"""
        # Set up wflow model data generation system
        wflow_data = WflowSimulationsGenerator(
            self.hydromt_path,
            self.hydro_combination_path,
            self.forcing_path,
            self.start_time,
            self.end_time,
            self.subbasin,
            self.strord,
            self.bbox,
            self.num_threads,
            self.polygons,
            self.resolution
        )
        
        # Generate wflow model data
        wflow_data.wflow_model_simulations_pipeline()
        
    def flood_data_pipeline(self):
        """Generate flood model data"""
        # Set up flood model data generation system
        flood_data = FloodModelSimulationsGenerator(
            self.hydro_combination_path,
            self.hydro_combination_path,
            self.precipitation_path,
            self.flood_aoi_boundary,
            self.start_time,
            self.end_time,
            self.crs,
            self.polygons,
            self.vectors
        )
        
        # Generate flood model data
        flood_data.flood_model_executor()
        
    def hydrological_and_hydrodynamic_simulation_generator(self):
        """Generate hydraulic and hydrodynamic simulations"""
        # The if function here will be modified later
        if self.polygons is not None:
            # Apply solutions
            self.total_solutions()

            # Generate wflow data
            self.wflow_data_pipeline()

            # Generate flood data
            self.flood_data_pipeline()

        elif self.vectors is not None:
            # Apply solutions
            self.total_solutions()

            # Generate flood data
            self.flood_data_pipeline()

        else:
            # Generate terrain data for wflow and flood models
            self.terrain_data_pipeline()

            # Generate wflow data
            self.wflow_data_pipeline()

            # Generate flood data
            self.flood_data_pipeline()


# This is where to check the model
def main():
    hydro_combination_path = Path(r"D:\Digital_Twin_data\hydrological_hydrodynamic_path_015")
    outlet_gauge_locations_filename = 'river_outlet'
    forcing_path = Path(r"H:\Barra\Whirinaki\merge_gauges_HIRDS_004")
    precipitation_path = Path(r"H:\Barra\Whirinaki\rainfall_gauges_HIRDS_004")
    start_time = datetime.fromisoformat("1999-01-20T00:00:00")
    end_time = datetime.fromisoformat("1999-01-22T12:00:00")

    # subbasin = [173.46365, -35.45662]
    # bbox = [173.44134, -35.61760, 173.77608, -35.11105]


    subbasin = [1642072.60, 6076218.85]
    bbox = [1639968.20, 6058374.30, 1670723.51, 6114366.30]

    num_threads = 6
    flood_aoi_boundary = [1641145.361, 6072406.885, 1642792.613, 6076268]

    polygons = r'polygons/polygons.shp' # r'polygons/polygons.shp'
    vectors = r'vectors/vectors.csv' # r'vectors/vectors.csv'
    strord = 4
    resolution = 50
    threshold = 1000
    width_rate_control = 1 / 20
    discharge_rate_control = 1
    crs = 2193

    # Set up hydraulic and hydrodynamic pipeline
    hydrological_hydrodynamic_pipeline = HydrologicalAndHydrodynamicPipeline(
        hydro_combination_path,
        outlet_gauge_locations_filename,

        forcing_path,
        precipitation_path,
        start_time,
        end_time,

        subbasin,
        bbox,
        num_threads,
        flood_aoi_boundary,

        polygons,
        vectors,
        strord,
        resolution,
        threshold,
        width_rate_control,
        discharge_rate_control,
        crs
    )

    hydrological_hydrodynamic_pipeline.hydrological_and_hydrodynamic_simulation_generator()

if __name__ == '__main__':
    main()