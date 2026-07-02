# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:11:15 2026

@author: mng42
"""
from osgeo import gdal
import logging
from datetime import datetime
import uuid
import os
import csv
from os import cpu_count
from pathlib import Path
from typing import Union

import geopandas as gpd
from shapely.geometry import box

from eddie.digitaltwin import retrieve_from_instructions
from eddie.digitaltwin.utils import setup_logging, LogLevel

from src.eddie_floodresilience.config import EnvVariable
from src.eddie_floodresilience.hydrological.wflow_serve_data_generator import WflowServeDataGenerator
from src.eddie_floodresilience.solutions.total_solutions import LandCoverSolution, ElevationSolution
from src.eddie_floodresilience.preprocessing.terrain_data_for_wflow_generator import TerrainDataWflowGenerator
from src.eddie_floodresilience.hydrological.wflow_simulations_generator import WflowSimulationsGenerator
from src.eddie_floodresilience.flood_model.bgflood.bgflood_simulations_generator import BGFloodModelSimulationsGenerator
from src.eddie_floodresilience.flood_model.lisflood.lisflood_simulations_generator import \
    LisFloodModelSimulationsGenerator

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
        flood_model: str,

        polygons: gpd.GeoDataFrame | None = None,
        vectors: str = None,
        resolution: float = 0.00045,
        threshold: int = 1000,
        landcover: str = 'globcover',
        folder_name: str = 'origin'
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
        folder_name: str = 'origin'
            Name of folder scenario
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
        self.folder_name = folder_name

        self.hydromt_path = EnvVariable.HYDROMT_PATH

        # River data
        self.river_name = river_name

        # Forcing data
        if isinstance(forcing_name, str) and not Path(forcing_name).exists():
            self.forcing_path = Path(self.hydromt_path / fr"forcing_data/{forcing_name}/forcing_*.nc")
        else:
            self.forcing_path = forcing_name

    def generate_folder_name(self) -> tuple[str, str]:
        """
        Generate folder name and type based on solution type

        Returns
        -------
        tuple[str, str]
            The folder name and type
        """
        if self.polygons is not None and self.vectors is not None:
            folder_name = "output_landcover_elevation"
            output_type = "landcover_elevation"
        elif self.polygons is not None:
            folder_name = "output_landcover"
            output_type = "landcover"
        elif self.vectors is not None:
            folder_name = "output_elevation"
            output_type = "elevation"
        else:
            folder_name = "output"
            output_type = "none"

    def reserve_pipeline_id(self) -> tuple[str, str]:
        """Generate ID simulation for the whole pipeline"""
        # Set folder ids list and path
        folder_ids = []
        folder_ids_path = self.hydro_combination_path / r'folder_id.csv'

        # Read the folder csv if it exists
        if os.path.exists(folder_ids_path):
            with open(folder_ids_path, 'r', newline='') as folder_text:
                reader_csv = csv.DictReader(folder_text)
                folder_ids = list(reader_csv)

            # Return existing ID if folder_name already there
            for folder_id in folder_ids:
                if folder_id['folder_name'] == self.folder_name:
                    return folder_id['model_id'], folder_id['folder_filename']

        # Generate new ID and save
        pipeline_id = str(uuid.uuid4())[:8]
        folder_filename = f"{self.folder_name}_{pipeline_id}"
        folder_ids.append({
            'folder_name': self.folder_name,
            'pipeline_id': pipeline_id,
            'folder_filename': folder_filename,
            'creation_time': datetime.now().isoformat()
        })

        # Write csv file
        fields_name = [
            'folder_name',
            'pipeline_id',
            'folder_filename',
            'creation_time'
        ]
        with open(folder_ids_path, 'w', newline='') as folder_text:
            writer_csv = csv.DictWriter(folder_text, fieldnames=fields_name)
            writer_csv.writeheader()
            writer_csv.writerows(folder_ids)

        return pipeline_id, folder_filename

    def total_solutions(self) -> None:
        """Develop solutions for flood risk resilience"""
        log.info("Starting total solutions")
        if self.polygons is not None and self.vectors is not None:
            # Land cover/natural solution
            landcover_solution = LandCoverSolution(
                self.hydro_combination_path,
                self.hydromt_path,
                self.polygons
            )
            self.landcover = landcover_solution.apply_landcover_solution().name

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
            self.landcover = landcover_solution.apply_landcover_solution().name

        elif self.vectors is not None:
            # Elevation solution
            elevation_solution = ElevationSolution(
                self.hydro_combination_path,
                self.flood_model,
                self.vectors
            )
            elevation_solution.apply_elevation_solution()

    def terrain_data_pipeline(self) -> None:
        """Generate terrain data for wflow and flood models"""
        log.info("Starting terrain data")
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

    def wflow_data_pipeline(self) -> None:
        """Generate wflow model data for flood model"""
        log.info("Starting wflow data pipeline")
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

    def serve_wflow_data(self, flood_model_output_id: int) -> None:
        """
        Serve data for a Wflow scenario, such as landcover and catchment boundaries.

        Parameters
        ----------
        flood_model_output_id: int
            The flood model output ID to associate the WFlow data with
        """
        log.info("Starting serve wflow data pipeline")
        wflow_serve_data = WflowServeDataGenerator(
            self.hydromt_path,
            self.hydro_combination_path,
            self.landcover,
            flood_model_output_id
        )

        wflow_serve_data.serve_data()

    def flood_data_pipeline(self) -> int:
        """
        Generate flood model data.

        Returns
        -------
        int
            The resultant flood model output ID.
        """
        log.info("Starting flood model pipeline")
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

        # The if function here will be modified later
        # Apply land cover solution
        # (this would be for both landcover and elevation solutions)
        if self.polygons is not None:
            # Apply solutions
            self.total_solutions()

            # Generate wflow data
            self.wflow_data_pipeline()

            # Generate flood data
            flood_model_output_id = self.flood_data_pipeline()

        # Apply elevation solution
        elif self.vectors is not None:
            # Apply solutions
            self.total_solutions()

            # Generate flood data
            flood_model_output_id = self.flood_data_pipeline()

        # Original scenario
        else:
            # # Generate terrain data for wflow and flood models
            # self.terrain_data_pipeline()

            # Generate wflow data
            self.wflow_data_pipeline()

            # Generate flood data
            flood_model_output_id = self.flood_data_pipeline()

        self.serve_wflow_data(flood_model_output_id)
        return flood_model_output_id


# OTAUTAU
def otautau(landcover_scenario_gdf: gpd.GeoDataFrame | None = None) -> int:
    """
    Run a hydrological and hydrodynamic simulation for Otautau.

    Parameters
    ----------
    landcover_scenario_gdf: gpd.GeoDataFrame | None
            Polygons that are used to change the landcover information.
            This polygon dataframe has 'landcover_name' column with new values.

    Returns
    -------
    int
        Flood model output ID.
    """
    hydro_combination_path = EnvVariable.HYDRO_COMBINATION_PATH_OTAUTAU
    forcing_name = 'otautau' # Path(r"H:/Barra/Mataura/merge_gauges_HIRDS_001")
    river_name = 'otautau'
    precipitation_path = Path(r"H:/Barra/Mataura/rainfall_gauges_HIRDS")
    start_time = datetime.fromisoformat("2020-02-03T00:00:00")
    end_time = datetime.fromisoformat("2020-02-05T00:00:00")

    # Gore
    num_threads = max(1, cpu_count() - 1)
    flood_aoi_boundary = [1211523.632, 4876273.859, 1215360.720, 4880693.039]
    adjust_manning = False
    flood_model = 'lisflood-fp'

    polygons = landcover_scenario_gdf # r'polygons/polygons.shp'
    vectors = None # r'vectors/vectors.csv'
    resolution = 200
    threshold = 25000
    landcover = 'lcdb_mapping'

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


def mataura(landcover_scenario_gdf: gpd.GeoDataFrame | None = None) -> int:
    """
    Run a hydrological and hydrodynamic simulation for Mataura.

    Parameters
    ----------
    landcover_scenario_gdf: gpd.GeoDataFrame | None
            Polygons that are used to change the landcover information.
            This polygon dataframe has 'landcover_name' column with new values.

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
    vectors = None  # r'vectors/vectors.csv'
    resolution = 200
    threshold = 25000
    landcover = 'lcdb_mapping'

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

    flood_model_output_id = hydrological_hydrodynamic_pipeline.hydrological_and_hydrodynamic_simulation_generator()
    return flood_model_output_id


def whirinaki(landcover_scenario_gdf: gpd.GeoDataFrame | None = None) -> int:
    """
    Run a hydrological and hydrodynamic simulation for Whirinaki.

    Parameters
    ----------
    landcover_scenario_gdf: gpd.GeoDataFrame | None
            Polygons that are used to change the landcover information.
            This polygon dataframe has 'landcover_name' column with new values.

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
    flood_model = 'lisflood-fp'

    polygons = landcover_scenario_gdf
    vectors = None  # r'vectors/vectors.csv'
    resolution = 50
    threshold = 1000
    landcover = 'lcdb_mapping'
    folder_name = 'changing_forest_001'

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
        landcover,
        folder_name
    )

    flood_model_output_id = hydrological_hydrodynamic_pipeline.hydrological_and_hydrodynamic_simulation_generator()
    return flood_model_output_id


# RIVERTON
def riverton(landcover_scenario_gdf: gpd.GeoDataFrame | None = None) -> int:
    """
    Run a hydrological and hydrodynamic simulation for Riverton.

    Parameters
    ----------
    landcover_scenario_gdf: gpd.GeoDataFrame | None
            Polygons that are used to change the landcover information.
            This polygon dataframe has 'landcover_name' column with new values.

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

    flood_model_output_id = hydrological_hydrodynamic_pipeline.hydrological_and_hydrodynamic_simulation_generator()
    return flood_model_output_id


if __name__ == '__main__':
    # Whirinaki
    # gdf = gpd.read_file(
    #     r"\\file\Research\DigitalTwins\smartideas\forLuke\automation_example"
    #     r"\polygons_vectors\whirinaki\polygons\polygons.shp"
    # )
    whirinaki(None)
    # whirinaki(gdf)

    # # Riverton
    # riverton(None)

    # # Mataura
    # gdf = gpd.read_file(
    #     r"D:\Digital_Twin_data\hydrological_hydrodynamic_path_031\mataura\polygons_upstream_thick\polygons.shp"
    # )
    # mataura(gdf)

    # gdf = gpd.read_file(
    #     r"D:\Digital_Twin_data\hydrological_hydrodynamic_path_031\otautau\polygons\polygons.shp"
    # )
    # otautau(gdf)
