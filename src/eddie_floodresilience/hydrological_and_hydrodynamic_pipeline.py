# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:11:15 2026

@author: mng42
"""

from pathlib import Path
from datetime import datetime
from typing import Union

import geopandas as gpd

from src.eddie_floodresilience.config import EnvVariable
from src.eddie_floodresilience.solutions.total_solutions import LandCoverSolution, ElevationSolution
from src.eddie_floodresilience.preprocessing.terrain_data_for_wflow_generator import TerrainDataWflowGenerator
from src.eddie_floodresilience.hydrological.wflow_simulations_generator import WflowSimulationsGenerator
from src.eddie_floodresilience.flood_model.bgflood.bgflood_simulations_generator import BGFloodModelSimulationsGenerator
from src.eddie_floodresilience.flood_model.lisflood.lisflood_simulations_generator import \
    LisFloodModelSimulationsGenerator

import logging
from eddie.digitaltwin.utils import setup_logging, LogLevel
setup_logging(LogLevel.DEBUG)
log = logging.getLogger(__name__)

class HydrologicalAndHydrodynamicPipeline:
    """This class is to generate hydrological and hydrodynamic results"""

    def __init__(
        self,
        hydro_combination_path: Path,

        forcing_name: Union[str, Path],
        river_name: Union[str, Path],
        precipitation_path: Path,
        start_time: datetime,
        end_time: datetime,

        num_threads: int,
        flood_aoi_boundary: list,
        adjust_manning: bool,
        flood_model: str,

        polygons: gpd.GeoDataFrame | None = None,
        vectors: str = None,
        resolution: float = 0.00045,
        threshold: int = 1000,
        landcover: str = 'globcover'
    ) -> None:
        """
        Generate hydrological and hydrodynamic results
        
        Parameters
        ----------
        hydro_combination_path : Path
            Directory to folder storing all necessary data
        forcing_name: Union[str, Path]
            Name of forcing data. Should be the site name. Ex: 'whirinaki'
            Or a directory to forcing data
        river_name: Union[str, Path]
            Name of river data. Should be the site name. Ex: 'whirinaki'
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
        num_threads : int
            Number of threads that controls how fast the wflow model can run
        flood_aoi_boundary : list
            Boundaries' coordinates of area of interest.
            Format is [xmin, ymin, xmax, ymax]
        adjust_manning : bool
            True means adjusting Manning's n by resampling 4m Manning's n
            False means no Mannning's n adjustment
        flood_model : str
            Either "lisflood-fp" or "bg-flood"

        polygons : gpd.GeoDataFrame | None = None
            Polygons that are used to change the landcover information.
            This polygon dataframe has 'landcover' column with new values
        vectors : str = None
            Name of vector file that is used to change the elevation information.
            This vector dataframe has 'value' column to specify increasing or decreasing elevation,
            and 'distance' column to specify how smooth to decrease elevation.
        resolution : float
            Resolution for flow data. 
            Default is 0.00045 (in crs 4326) ~ 50 m (in crs 2193)
        threshold: int = 1000
            Minimum number of cells/up-slope area required to initiate and main a channel.
            Default is 1000
        landcover: str = 'globcover'
            Name of land cover dataset. Default is 'globcover'
        """
        # Set up necessary parameters
        self.hydro_combination_path = hydro_combination_path
        self.precipitation_path = precipitation_path
        self.start_time = start_time
        self.end_time = end_time
        self.num_threads = num_threads
        self.flood_aoi_boundary = flood_aoi_boundary
        self.adjust_manning = adjust_manning
        self.flood_model = flood_model

        self.polygons = polygons
        self.vectors = vectors
        self.resolution = resolution
        self.threshold = threshold
        self.crs = 2193
        self.landcover = landcover

        self.hydromt_path = EnvVariable.HYDROMT_PATH

        # River data
        self.river_name = river_name

        # Forcing data
        if isinstance(forcing_name, str) and not Path(forcing_name).exists():
            self.forcing_path = Path(self.hydromt_path / fr"forcing_data/{forcing_name}/forcing_*.nc")
        else:
            self.forcing_path = forcing_name

    def total_solutions(self):
        print("Starting total solutions")
        """Develop solutions for flood risk resilience"""
        if self.polygons is not None and self.vectors is not None:
            # Land cover/natural solution
            landcover_solution = LandCoverSolution(
                self.hydro_combination_path,
                self.hydromt_path,
                self.polygons
            )
            landcover_solution.apply_landcover_solution()

            # Elevation solution
            elevation_solution = ElevationSolution(
                self.hydro_combination_path,
                self.flood_model,
                self.vectors
            )
            elevation_solution.apply_elevation_solution()

        elif self.polygons is not None:
            # Land cover/natural solution
            landcover_solution = LandCoverSolution(
                self.hydro_combination_path,
                self.hydromt_path,
                self.polygons
            )
            landcover_solution.apply_landcover_solution()

        elif self.vectors is not None:
            # Elevation solution
            elevation_solution = ElevationSolution(
                self.hydro_combination_path,
                self.flood_model,
                self.vectors
            )
            elevation_solution.apply_elevation_solution()

    def terrain_data_pipeline(self):
        """Generate terrain data for wflow and flood models"""
        print("Starting terrain data")
        # Set up terrain data generation system
        terrain_data = TerrainDataWflowGenerator(
            self.hydro_combination_path,
            self.hydromt_path,
            self.flood_aoi_boundary,
            self.river_name,
            self.resolution,
            self.threshold
        )

        # Generate terrain data
        terrain_data.terrain_for_wflow_generator()

    def wflow_data_pipeline(self):
        """Generate wflow model data for flood model"""
        print("Starting wflow data pipeline")
        # Set up wflow model data generation system
        wflow_data = WflowSimulationsGenerator(
            self.hydromt_path,
            self.hydro_combination_path,
            self.river_name,
            self.forcing_path,
            self.start_time,
            self.end_time,
            self.flood_aoi_boundary,
            self.num_threads,
            self.polygons,
            self.resolution,
            self.landcover
        )

        # Generate wflow model data
        wflow_data.wflow_model_simulations_pipeline()

    def flood_data_pipeline(self):
        """Generate flood model data"""
        print("Starting flood model pipeline")
        # Set up flood model data generation system
        if self.flood_model == 'lisflood-fp':
            flood_data = LisFloodModelSimulationsGenerator(
                self.hydro_combination_path,
                self.hydro_combination_path,
                self.hydromt_path,
                self.river_name,
                self.precipitation_path,
                self.flood_aoi_boundary,
                self.adjust_manning,
                self.start_time,
                self.end_time,
                self.crs,
                self.polygons,
                self.vectors
            )

        else:
            flood_data = BGFloodModelSimulationsGenerator(
                self.hydro_combination_path,
                self.hydro_combination_path,
                self.hydromt_path,
                self.river_name,
                self.precipitation_path,
                self.flood_aoi_boundary,
                self.adjust_manning,
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
        self.hydro_combination_path.mkdir(parents=True, exist_ok=True)
        # The if function here will be modified later
        # Apply land cover solution
        # (this would be for both landcover and elevation solutions)
        if self.polygons is not None:
            # Apply solutions
            self.total_solutions()

            # Generate wflow data
            self.wflow_data_pipeline()

            # Generate flood data
            self.flood_data_pipeline()

        # Apply elevation solution
        elif self.vectors is not None:
            # Apply solutions
            self.total_solutions()

            # Generate flood data
            self.flood_data_pipeline()

        # Original scenario
        else:
            # Generate terrain data for wflow and flood models
            self.terrain_data_pipeline()

            # Generate wflow data
            self.wflow_data_pipeline()

            # Generate flood data
            self.flood_data_pipeline()


# # OTAUTAU
# def main():
#     hydro_combination_path = Path(r"D:/Digital_Twin_data/hydrological_hydrodynamic_otautau_path_009")
#     forcing_name = Path(r"H:/Barra/Mataura/merge_gauges_HIRDS_001") # Path(r"H:/Barra/Mataura/merge_gauges_HIRDS_001")
#     river_name = 'otautau'
#     precipitation_path = Path(r"H:/Barra/Mataura/rainfall_gauges_HIRDS")
#     start_time = datetime.fromisoformat("2020-02-03T00:00:00")
#     end_time = datetime.fromisoformat("2020-02-05T00:00:00")
#
#     # Gore
#     num_threads = 8
#     flood_aoi_boundary = [1211523.632, 4876273.859, 1215360.720, 4880693.039]
#     adjust_manning = False
#     flood_model = 'bg-flood'
#
#     polygons = None # r'polygons/polygons.shp'
#     vectors = None # r'vectors/vectors.csv'
#     resolution = 200
#     threshold = 25000
#     landcover = 'globcover'
#
#     # Set up hydraulic and hydrodynamic pipeline
#     hydrological_hydrodynamic_pipeline = HydrologicalAndHydrodynamicPipeline(
#         hydro_combination_path,
#
#         forcing_name,
#         river_name,
#         precipitation_path,
#         start_time,
#         end_time,
#
#         num_threads,
#         flood_aoi_boundary,
#         adjust_manning,
#         flood_model,
#
#         polygons,
#         vectors,
#         resolution,
#         threshold,
#         landcover
#     )
#
#     hydrological_hydrodynamic_pipeline.hydrological_and_hydrodynamic_simulation_generator()
#
# if __name__ == '__main__':
#     main()


# # WAIMEA
# def main():
#     hydro_combination_path = Path(r"D:/Digital_Twin_data/hydrological_hydrodynamic_waimea_path_005")
#     forcing_name = Path(r"H:/Barra/Mataura/merge_gauges_HIRDS_001") # Path(r"H:/Barra/Mataura/merge_gauges_HIRDS_001")
#     river_name = 'waimea'
#     precipitation_path = Path(r"H:/Barra/Mataura/rainfall_gauges_HIRDS")
#     start_time = datetime.fromisoformat("2020-02-03T00:00:00")
#     end_time = datetime.fromisoformat("2020-02-05T00:00:00")
#
#     # Gore
#     num_threads = 8
#     flood_aoi_boundary = [1274171.417, 4896113.968, 1279068.442, 4899628.700]
#     adjust_manning = False
#     flood_model = 'lisflood-fp'
#
#     polygons = None # r'polygons/polygons.shp'
#     vectors = None # r'vectors/vectors.csv'
#     resolution = 200
#     threshold = 25000
#     landcover = 'globcover'
#
#     # Set up hydraulic and hydrodynamic pipeline
#     hydrological_hydrodynamic_pipeline = HydrologicalAndHydrodynamicPipeline(
#         hydro_combination_path,
#
#         forcing_name,
#         river_name,
#         precipitation_path,
#         start_time,
#         end_time,
#
#         num_threads,
#         flood_aoi_boundary,
#         adjust_manning,
#         flood_model,
#
#         polygons,
#         vectors,
#         resolution,
#         threshold,
#         landcover
#     )
#
#     hydrological_hydrodynamic_pipeline.hydrological_and_hydrodynamic_simulation_generator()
#
# if __name__ == '__main__':
#     main()


# MATAURA
def main(landcover_scenario_gdf: gpd.GeoDataFrame | None = None):
    print("main started")

    hydro_combination_path = EnvVariable.HYDRO_COMBINATION_PATH
    forcing_name = 'mataura'  # Path(r"H:/Barra/Mataura/merge_gauges_HIRDS_001")
    river_name = 'mataura'
    precipitation_path = EnvVariable.PRECIPITATION_PATH
    start_time = datetime.fromisoformat("2020-02-03T00:00:00")
    end_time = datetime.fromisoformat("2020-02-05T00:00:00")

    # Gore
    num_threads = 8
    flood_aoi_boundary = [1283763.983, 4882997.604, 1289535.012, 4890957.772]
    adjust_manning = False
    flood_model = 'lisflood-fp'

    polygons = None
    vectors = None  # r'vectors/vectors.csv'
    resolution = 200
    threshold = 25000
    landcover = 'globcover'

    # Set up hydraulic and hydrodynamic pipeline
    hydrological_hydrodynamic_pipeline = HydrologicalAndHydrodynamicPipeline(
        hydro_combination_path,

        forcing_name,
        river_name,
        precipitation_path,
        start_time,
        end_time,

        num_threads,
        flood_aoi_boundary,
        adjust_manning,
        flood_model,

        polygons,
        vectors,
        resolution,
        threshold,
        landcover
    )

    hydrological_hydrodynamic_pipeline.hydrological_and_hydrodynamic_simulation_generator()


# WHIRINAKI
# This is where to check the model
def whirinaki():
    hydro_combination_path = EnvVariable.HYDRO_COMBINATION_PATH
    forcing_name = 'whirinaki'
    river_name = 'whirinaki'
    precipitation_path = EnvVariable.PRECIPITATION_PATH
    start_time = datetime.fromisoformat("1999-01-20T00:00:00")
    end_time = datetime.fromisoformat("1999-01-22T12:00:00")

    num_threads = 8
    flood_aoi_boundary = [1641148, 6072404, 1642796, 6076268]
    adjust_manning = True
    flood_model = 'bg-flood'

    polygons = None  # r'polygons/polygons.shp'
    vectors = None  # r'vectors/vectors.csv'
    resolution = 50
    threshold = 1000
    landcover = 'globcover'

    # Set up hydraulic and hydrodynamic pipeline
    hydrological_hydrodynamic_pipeline = HydrologicalAndHydrodynamicPipeline(
        hydro_combination_path,

        forcing_name,
        river_name,
        precipitation_path,
        start_time,
        end_time,

        num_threads,
        flood_aoi_boundary,
        adjust_manning,
        flood_model,

        polygons,
        vectors,
        resolution,
        threshold,
        landcover
    )

    hydrological_hydrodynamic_pipeline.hydrological_and_hydrodynamic_simulation_generator()

if __name__ == '__main__':
    whirinaki()