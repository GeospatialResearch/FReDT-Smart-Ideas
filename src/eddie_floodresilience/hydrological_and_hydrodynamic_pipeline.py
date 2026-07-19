# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:11:15 2026

@author: mng42
"""

import logging
from datetime import datetime
from os import cpu_count
from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from sqlalchemy import insert

from eddie.digitaltwin import retrieve_from_instructions
from eddie.digitaltwin.setup_environment import get_database
from eddie.digitaltwin.tables import create_table
from eddie.digitaltwin.utils import setup_logging, LogLevel

from src.eddie_floodresilience.config import EnvVariable
from src.eddie_floodresilience.hydrological.wflow_data_catalog_generator import DataCatalogGenerator
from src.eddie_floodresilience.hydrological.wflow_serve_data_generator import WflowServeDataGenerator
from src.eddie_floodresilience.solutions.total_solutions import LandCoverSolution, ElevationSolution
from src.eddie_floodresilience.preprocessing.terrain_data_for_wflow_generator import TerrainDataWflowGenerator
from src.eddie_floodresilience.hydrological.wflow_simulations_generator import WflowSimulationsGenerator
from src.eddie_floodresilience.flood_model.bgflood.bgflood_simulations_generator import BGFloodModelSimulationsGenerator
from src.eddie_floodresilience.flood_model.lisflood.lisflood_simulations_generator import \
    LisFloodModelSimulationsGenerator
from src.eddie_floodresilience.tables import PipelineOutput

setup_logging(LogLevel.DEBUG)
log = logging.getLogger(__name__)


class HydrologicalAndHydrodynamicPipeline:
    """
    This class is to generate hydrological and hydrodynamic results

    Attributes
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
        flood_type : str = 'fluvial'
            Flood type: 'pluvial' or 'fluvial'. Default is 'fluvial'
        polygons : gpd.GeoDataFrame | None = None
            Polygons that are used to change the landcover information.
            This polygon dataframe has 'landcover' column with new values
        vectors : pd.DataFrame = None
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
    """  # pylint: disable=too-many-instance-attributes

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

        flood_model: str = 'lisflood-fp',
        flood_type: str = 'fluvial',

        polygons: gpd.GeoDataFrame | None = None,
        vectors: pd.DataFrame = None,
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
            Name of forcing data. Should be the site name. Ex: 'whirin
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
        flood_model : str = 'lisflood-fp'
            Either "lisflood-fp" or "bg-flood"
        flood_type : str = 'fluvial'
            Flood type: 'pluvial' or 'fluvial'. Default is 'fluvial'
        polygons : gpd.GeoDataFrame | None = None
            Polygons that are used to change the landcover information.
            This polygon dataframe has 'landcover' column with new values
        vectors : pd.DataFrame = None
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
        self.flood_type = flood_type

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

    def reserve_scenario_id(self) -> int:
        """
        Generate and reserve scenario ID in the database.

        Returns
        -------
        int
            The new scenario ID
        """
        engine = get_database()
        with engine.connect() as conn:
            # Create the scenario table if not exist
            create_table(conn, PipelineOutput)

            # Create a query to reserve scenario ID
            query = insert(PipelineOutput).values(geometry=box(*self.flood_aoi_boundary).wkt)

            # Execute the query
            result = conn.execute(query)

        # Set up scenario ID
        scenario_id = result.inserted_primary_key[0]

        # Log a message
        log.info(f"Scenario ID {scenario_id} reserved.")

        return scenario_id

    def scenario_folder_generator(self) -> Path:
        """
        Generate scenario folder based on solutions

        Returns
        -------
        Path
            Directory to the scenario folder name with ID
        """
        # Set up log message
        if self.polygons is None and self.vectors is None:
            log.info("Generating origin folder")
        else:
            log.info("Generating scenario folder")

        # Set up scenario id
        scenario_id = self.reserve_scenario_id()

        # Choose scenario folder name with ID
        if self.polygons is not None and self.vectors is not None:
            scenario_and_id_name = f"scenario_landcover_elevation_{scenario_id}"

            # Create hydrological process path
            hydrological_process_path = self.hydro_combination_path / scenario_and_id_name / 'hydrological_process'
            hydrological_process_path.mkdir(parents=True, exist_ok=True)

        elif self.polygons is not None:
            scenario_and_id_name = f"scenario_landcover_{scenario_id}"

        elif self.vectors is not None:
            scenario_and_id_name = f"scenario_elevation_{scenario_id}"

        else:
            scenario_and_id_name = "original_scenario"

            # Create hydrological process path
            hydrological_process_path = self.hydro_combination_path / scenario_and_id_name / 'hydrological_process'
            hydrological_process_path.mkdir(parents=True, exist_ok=True)

        # Create hydrological process path
        hydrodynamic_process_path = self.hydro_combination_path / scenario_and_id_name / 'hydrodynamic_process'
        hydrodynamic_process_path.mkdir(parents=True, exist_ok=True)

        # Set up scenario
        scenario_and_id_folder = self.hydro_combination_path / scenario_and_id_name

        return scenario_and_id_folder

    def total_solutions(
            self,
            scenario_and_id_folder: Path
    ) -> None:
        """
        Develop solutions for flood risk resilience

        Parameters
        ----------
        scenario_and_id_folder : Path
            Directory to the scenario folder name with ID
        """
        # Set up log message
        log.info("Starting total solutions")

        if self.polygons is not None and self.vectors is not None:

            # Land cover/natural solution
            landcover_solution = LandCoverSolution(
                self.hydromt_path,
                scenario_and_id_folder,
                self.landcover,
                self.polygons
            )
            self.landcover = landcover_solution.apply_landcover_solution().name

            # Elevation solution
            elevation_solution = ElevationSolution(
                self.flood_model,
                scenario_and_id_folder,
                self.vectors
            )
            elevation_solution.apply_elevation_solution()

        elif self.polygons is not None:

            # Land cover/natural solution
            landcover_solution = LandCoverSolution(
                self.hydromt_path,
                scenario_and_id_folder,
                self.landcover,
                self.polygons
            )
            self.landcover = landcover_solution.apply_landcover_solution().name

        elif self.vectors is not None:

            # Elevation solution
            elevation_solution = ElevationSolution(
                self.flood_model,
                scenario_and_id_folder,
                self.vectors
            )
            elevation_solution.apply_elevation_solution()

    def terrain_data_pipeline(self) -> None:
        """Generate terrain data for wflow and flood models"""
        log.info("Starting terrain data")

        # Set up terrain folder
        terrain_folder = self.hydro_combination_path / 'terrain'
        terrain_folder.mkdir(parents=True, exist_ok=True)

        # Set up terrain data generation system
        terrain_data = TerrainDataWflowGenerator(
            terrain_folder,
            self.hydromt_path,
            self.flood_aoi_boundary,
            self.river_name,
            self.resolution,
            self.threshold
        )

        # Generate terrain data
        terrain_data.terrain_for_wflow_generator()

    def wflow_data_pipeline(
            self,
            scenario_and_id_folder: Path
    ) -> None:
        """
        Generate wflow model data for flood model

        Parameters
        ----------
        scenario_and_id_folder : Path
            Directory to the scenario folder name with ID
        """
        log.info("Starting wflow data pipeline")

        # Set up wflow model data generation system
        wflow_data = WflowSimulationsGenerator(
            self.hydromt_path,
            self.river_name,
            self.forcing_path,
            self.start_time,
            self.end_time,
            self.flood_aoi_boundary,
            self.num_threads,
            scenario_and_id_folder,
            self.polygons,
            self.resolution,
            self.landcover
        )

        # Generate wflow model data
        wflow_data.wflow_model_simulations_pipeline()

    def serve_wflow_data(self, scenario_and_id_folder: Path, flood_model_output_id: int) -> None:
        """
        Serve data for a Wflow scenario, such as landcover and catchment boundaries.

        Parameters
        ----------
        scenario_and_id_folder: Path
            Directory to the scenario folder name with ID
        flood_model_output_id: int
            The flood model output ID to associate the WFlow data with
        """
        landcover_mapping_type = DataCatalogGenerator.landcover_mapping_type(self.landcover)
        wflow_serve_data = WflowServeDataGenerator(
            self.hydromt_path,
            self.polygons,
            landcover_mapping_type,
            scenario_and_id_folder,
            flood_model_output_id
        )

        wflow_serve_data.serve_data()

    def flood_data_pipeline(
            self,
            scenario_and_id_folder: Path
    ) -> int:
        """
        Generate flood model data.

        Parameters
        ----------
        scenario_and_id_folder : Path
            Directory to the scenario folder name with ID

        Returns
        -------
        int
            The resultant flood model output ID.
        """
        log.info("Starting flood model pipeline")

        # Set up flood model data generation system
        if self.flood_model == 'lisflood-fp':
            flood_data = LisFloodModelSimulationsGenerator(
                self.hydromt_path,
                self.river_name,
                self.precipitation_path,
                self.flood_aoi_boundary,
                self.adjust_manning,
                self.start_time,
                self.end_time,
                scenario_and_id_folder,
                self.flood_type,
                self.crs,
                self.polygons,
                self.vectors
            )

        else:
            # pylint: disable=E1121
            flood_data = BGFloodModelSimulationsGenerator(
                self.hydromt_path,
                self.river_name,
                self.precipitation_path,
                self.flood_aoi_boundary,
                self.adjust_manning,
                self.start_time,
                self.end_time,
                scenario_and_id_folder,
                self.flood_type,
                self.crs,
                self.polygons,
                self.vectors
            )

        # Generate flood model data
        model_output_id = flood_data.flood_model_executor()
        return model_output_id

    def hydrological_and_hydrodynamic_simulation_generator(self) -> int:
        """
        Generate hydraulic and hydrodynamic simulations.

        Returns
        -------
        int
            The resultant flood model output ID.
        """
        # Ensure output directory exists
        self.hydro_combination_path.mkdir(parents=True, exist_ok=True)

        # Download external spatial data into database
        bbox_gdf = gpd.GeoDataFrame(geometry=[box(*self.flood_aoi_boundary)], crs="EPSG:2193")
        retrieve_from_instructions.main(bbox_gdf, Path("src/eddie_floodresilience/static_boundary_instructions.json"))

        # Set scenario
        scenario_and_id_folder = self.scenario_folder_generator()

        # The if function here will be modified later
        # Apply land cover solution
        # (this would be for both landcover and elevation solutions)
        if self.polygons is not None:
            # Apply solutions
            self.total_solutions(scenario_and_id_folder)

            # Generate wflow data
            self.wflow_data_pipeline(scenario_and_id_folder)

            # Generate flood data
            flood_model_output_id = self.flood_data_pipeline(scenario_and_id_folder)

        # Apply elevation solution
        elif self.vectors is not None:
            # Apply solutions
            self.total_solutions(scenario_and_id_folder)

            # Generate flood data
            flood_model_output_id = self.flood_data_pipeline(scenario_and_id_folder)

        # Original scenario
        else:
            # Generate terrain data for wflow and flood models
            self.terrain_data_pipeline()

            # Generate wflow data
            self.wflow_data_pipeline(scenario_and_id_folder)

            # Generate flood data
            flood_model_output_id = self.flood_data_pipeline(scenario_and_id_folder)

        self.serve_wflow_data(scenario_and_id_folder, flood_model_output_id)
        return flood_model_output_id


# OTAUTAU
def otautau(
        flood_type: str = 'fluvial',
        landcover_scenario_gdf: gpd.GeoDataFrame | None = None,
        elevation_scenario_df: pd.DataFrame | None = None
) -> int:
    """
    Run a hydrological and hydrodynamic simulation for Otautau.

    Parameters
    ----------
    flood_type : str = 'fluvial'
        Flood type: 'pluvial' or 'fluvial'. Default is 'fluvial'
    landcover_scenario_gdf: gpd.GeoDataFrame | None
        Polygons that are used to change the landcover information.
        This polygon dataframe has 'landcover_name' column with new values.
    elevation_scenario_df: pd.DataFrame | None
        Dataframe that contains 'vector_path', 'value', 'distance' columns:
        - 'vector_path': Column that stores directories to specific vectors
        - 'value: Column that stores value of the vectors used to increase/decrease elevation
        - 'distance': Column that stores value to smooth the decreased elevation

    Returns
    -------
    int
        Flood model output ID.
    """
    hydro_combination_path = EnvVariable.HYDRO_COMBINATION_PATH_OTAUTAU
    forcing_name = 'otautau'  # Path(r"H:/Barra/Mataura/merge_gauges_HIRDS_001")
    river_name = 'otautau'
    precipitation_path = Path(r"H:/Barra/Mataura/rainfall_gauges_HIRDS")
    start_time = datetime.fromisoformat("2020-02-03T00:00:00")
    end_time = datetime.fromisoformat("2020-02-05T00:00:00")

    # Gore
    num_threads = max(1, cpu_count() - 1)
    flood_aoi_boundary = [1211523.632, 4876273.859, 1215360.720, 4880693.039]
    adjust_manning = False
    flood_model = 'lisflood-fp'

    polygons = landcover_scenario_gdf  # r'polygons/polygons.shp'
    vectors = elevation_scenario_df  # r'vectors/vectors.csv'
    resolution = 200
    threshold = 25000
    landcover = 'lcdb'

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
        flood_type,

        polygons,
        vectors,
        resolution,
        threshold,
        landcover
    )

    flood_model_output_id = hydrological_hydrodynamic_pipeline.hydrological_and_hydrodynamic_simulation_generator()
    return flood_model_output_id

# # WAIMEA
# def main():
#     hydro_combination_path = Path(r"D:/Digital_Twin_data/hydrological_hydrodynamic_waimea_path_027")
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


def mataura(
        flood_type: str = 'fluvial',
        landcover_scenario_gdf: gpd.GeoDataFrame | None = None,
        elevation_scenario_df: pd.DataFrame | None = None
) -> int:
    """
    Run a hydrological and hydrodynamic simulation for Otautau.

    Parameters
    ----------
    flood_type : str = 'fluvial'
        Flood type: 'pluvial' or 'fluvial'. Default is 'fluvial'
    landcover_scenario_gdf: gpd.GeoDataFrame | None
        Polygons that are used to change the landcover information.
        This polygon dataframe has 'landcover_name' column with new values.
    elevation_scenario_df: pd.DataFrame | None
        Dataframe that contains 'vector_path', 'value', 'distance' columns:
        - 'vector_path': Column that stores directories to specific vectors
        - 'value: Column that stores value of the vectors used to increase/decrease elevation
        - 'distance': Column that stores value to smooth the decreased elevation

    Returns
    -------
    int
        Flood model output ID.
    """
    hydro_combination_path = EnvVariable.HYDRO_COMBINATION_PATH_MATAURA
    forcing_name = 'mataura'  # Path(r"H:/Barra/Mataura/merge_gauges_HIRDS_001")
    river_name = 'mataura'
    precipitation_path = EnvVariable.PRECIPITATION_PATH / "mataura"
    start_time = datetime.fromisoformat("2020-02-03T00:00:00")
    end_time = datetime.fromisoformat("2020-02-05T00:00:00")

    # Gore
    num_threads = max(1, cpu_count() - 1)
    flood_aoi_boundary = [1283763.983, 4882997.604, 1289535.012, 4890957.772]
    adjust_manning = False
    flood_model = 'lisflood-fp'

    polygons = landcover_scenario_gdf
    vectors = elevation_scenario_df  # r'vectors/vectors.csv'
    resolution = 200
    threshold = 25000
    landcover = 'lcdb'

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
        flood_type,

        polygons,
        vectors,
        resolution,
        threshold,
        landcover
    )

    flood_model_output_id = hydrological_hydrodynamic_pipeline.hydrological_and_hydrodynamic_simulation_generator()
    return flood_model_output_id


def whirinaki(
        flood_type: str = 'fluvial',
        landcover_scenario_gdf: gpd.GeoDataFrame | None = None,
        elevation_scenario_df: pd.DataFrame | None = None
) -> int:
    """
    Run a hydrological and hydrodynamic simulation for Otautau.

    Parameters
    ----------
    flood_type : str = 'fluvial'
        Flood type: 'pluvial' or 'fluvial'. Default is 'fluvial'
    landcover_scenario_gdf: gpd.GeoDataFrame | None
        Polygons that are used to change the landcover information.
        This polygon dataframe has 'landcover_name' column with new values.
    elevation_scenario_df: pd.DataFrame | None
        Dataframe that contains 'vector_path', 'value', 'distance' columns:
        - 'vector_path': Column that stores directories to specific vectors
        - 'value: Column that stores value of the vectors used to increase/decrease elevation
        - 'distance': Column that stores value to smooth the decreased elevation

    Returns
    -------
    int
        Flood model output ID.
    """
    hydro_combination_path = EnvVariable.HYDRO_COMBINATION_PATH_WHIRINAKI
    forcing_name = 'whirinaki'
    river_name = 'whirinaki'
    precipitation_path = EnvVariable.PRECIPITATION_PATH / "whirinaki"
    start_time = datetime.fromisoformat("1999-01-20T00:00:00")
    end_time = datetime.fromisoformat("1999-01-22T12:00:00")

    num_threads = max(1, cpu_count() - 1)
    flood_aoi_boundary = [1641148, 6072532, 1642796, 6076268]
    adjust_manning = False
    flood_model = 'bg-flood'

    polygons = landcover_scenario_gdf
    vectors = elevation_scenario_df  # r'vectors/vectors.csv'
    resolution = 50
    threshold = 1000
    landcover = 'lcdb'

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
        flood_type,

        polygons,
        vectors,
        resolution,
        threshold,
        landcover
    )

    flood_model_output_id = hydrological_hydrodynamic_pipeline.hydrological_and_hydrodynamic_simulation_generator()
    return flood_model_output_id


# RIVERTON
def riverton(
        flood_type: str = 'fluvial',
        landcover_scenario_gdf: gpd.GeoDataFrame | None = None,
        elevation_scenario_df: pd.DataFrame | None = None
) -> int:
    """
    Run a hydrological and hydrodynamic simulation for Otautau.

    Parameters
    ----------
    flood_type : str = 'fluvial'
        Flood type: 'pluvial' or 'fluvial'. Default is 'fluvial'
    landcover_scenario_gdf: gpd.GeoDataFrame | None
        Polygons that are used to change the landcover information.
        This polygon dataframe has 'landcover_name' column with new values.
    elevation_scenario_df: pd.DataFrame | None
        Dataframe that contains 'vector_path', 'value', 'distance' columns:
        - 'vector_path': Column that stores directories to specific vectors
        - 'value: Column that stores value of the vectors used to increase/decrease elevation
        - 'distance': Column that stores value to smooth the decreased elevation

    Returns
    -------
    int
        Flood model output ID.
    """
    hydro_combination_path = EnvVariable.HYDRO_COMBINATION_PATH_RIVERTON
    forcing_name = 'riverton'  # Forcing data is already pre-processed
    river_name = 'riverton'
    precipitation_path = EnvVariable.PRECIPITATION_PATH / "mataura"  # Mataura data is for Southland
    start_time = datetime.fromisoformat("2020-02-03T00:00:00")
    end_time = datetime.fromisoformat("2020-02-05T00:00:00")

    num_threads = max(1, cpu_count() - 1)
    flood_aoi_boundary = [1209555.319, 4849977.393, 1222804.726, 4864906.303]
    adjust_manning = False
    flood_model = 'lisflood-fp'

    polygons = landcover_scenario_gdf
    vectors = elevation_scenario_df  # r'vectors/vectors.csv'
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
        flood_type,

        polygons,
        vectors,
        resolution,
        threshold,
        landcover
    )

    flood_model_output_id = hydrological_hydrodynamic_pipeline.hydrological_and_hydrodynamic_simulation_generator()
    return flood_model_output_id


if __name__ == '__main__':
    # Whirinaki
    gdf = gpd.read_file(
        r"D:\Digital_Twin_data\hydrological_hydrodynamic_path_031\whirinaki\polygons\polygons.shp"
    )
    df = pd.read_csv(
        r"D:\Digital_Twin_data\hydrological_hydrodynamic_path_031\whirinaki\vectors\vectors.csv"
    )
    whirinaki('fluvial', None, None)

    # # Riverton
    # riverton(None, None)

    # # Mataura
    # gdf = gpd.read_file(
    #     r"D:\Digital_Twin_data\hydrological_hydrodynamic_path_031\mataura\polygons_upstream_thick\polygons.shp"
    # )
    # mataura(gdf, None)

    # # Otautau
    # gdf = gpd.read_file(
    #     r"D:\Digital_Twin_data\hydrological_hydrodynamic_path_031\otautau\polygons\polygons.shp"
    # )
    # otautau(gdf, None)
