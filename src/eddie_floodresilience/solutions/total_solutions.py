import logging

from osgeo import gdal
import rioxarray as rxr
import xarray as xr
from pathlib import Path
import geopandas as gpd
import pandas as pd
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
import numpy as np

from whitebox_workflows import WbEnvironment, Raster
from whitebox.whitebox_tools import WhiteboxTools

from eddie.digitaltwin.utils import setup_logging, LogLevel

setup_logging(LogLevel.DEBUG)
log = logging.getLogger(__name__)

GLOBCOVER_CLASSES: dict[str, int] = {
    "Dense Deciduous Forest": 50,
    "Evergreen Forest": 40,
    "Deciduous Forest": 60,
    "Needleleaf Forest": 70,
    "Pasture Mosaic": 120,
    "Shrubland": 130,
    "Pasture": 140,
    "Sparse Vegetation": 150,
    "Wetland": 160,
    "Bare Land": 200,
}


wbe = WbEnvironment()
wbe.verbose = True
wbe.max_procs = -1

wbt = WhiteboxTools()


class LandCoverSolution():
    """This class is to change the land cover based on polygons"""

    def __init__(
            self,
            hydro_combination_path: Path,
            hydromt_path: Path,
            polygons: gpd.GeoDataFrame | None = None
    ) -> None:
        """
        Change the land cover based on polygons.
        This class relates to functions:
        - landcover_section in wflow_data_catalog_generator.py
        - par_generator in lisflood_parameters_generator.py
        - hydrological_and_hydrodynamic_simulation_generator in hydrological_and_hydrodynamic_pipeline.py

        Parameters
        ----------
        hydro_combination_path : Path
            Directory to folder storing all necessary data
        hydromt_path : Path
            A directory to where all necessary files are stored to run wflow model
        polygons : gpd.GeoDataFrame = None
            Polygons that are used to change the landcover information
        """
        self.hydro_combination_path = hydro_combination_path
        self.hydromt_path = hydromt_path
        self.polygons = polygons

    def rasterize_polygons(
            self,
            current_landcover: xr.DataArray,
            polygons: gpd.GeoDataFrame
    ) -> xr.DataArray:
        """
        This is to apply values to each polygon under raster format

        Parameters
        ----------
        current_landcover : xr.DataArray
            Raster of current land cover from LCDB-converted Global cover
        polygons : gpd.GeoDataFrame
            Polygons that are used to change the landcover information.
            This polygon dataframe has 'landcover' column with new values

        Returns
        -------
        modified_landcover : xr.DataArray
            Raster of land cover that is modified
        """
        # Copy original land cover data to not be affected by the change
        modified_landcover = current_landcover.copy()
        if "landcover" not in polygons.columns:
            polygons["landcover"] = polygons["landcover_name"].map(GLOBCOVER_CLASSES)
        
        # Create rasterization shapes
        shapes = [
            (geom, value)
            for geom, value in zip(
                polygons.geometry,
                polygons["landcover"]
            )
        ]

        # Rasterize all polygons at once
        polygon_raster = rasterize(
            shapes=shapes,
            out_shape=current_landcover.shape,
            transform=current_landcover.rio.transform(),
            fill=0,
            dtype='uint8'
        )

        # Applyg changes
        mask = polygon_raster != 0
        modified_landcover.values[mask] = polygon_raster[mask]

        return modified_landcover


    def apply_landcover_solution(self) -> Path:
        """
        Changes the landcover based on polygons.

        Returns
        -------
        Path
            Path to the modified landcover.
        """
        # Read current land cover data
        with rxr.open_rasterio(self.hydromt_path / r'original_globcover.tif') as current_landcover:
            current_landcover = current_landcover.squeeze().load()

        # Convert crs
        # This step will be removed in future
        polygons_crs = self.polygons.to_crs(4326)

        # Rasterize and apply new values to current land cover
        modified_landcover = self.rasterize_polygons(
            current_landcover,
            polygons_crs
        )

        # self.hydromt_path may not be writable on linux, so write to self.hydro_combination_path
        globcover_dir = self.hydro_combination_path / "globcover"
        globcover_dir.mkdir(parents=True, exist_ok=True)

        # Find existing elevation files
        existing_landcover_files = list(
            globcover_dir.glob("globcover_*.tif")
        )

        # Decide the number of output file
        # based on the number of running the model
        number = len(existing_landcover_files) + 1

        # Create filename

        output_path = globcover_dir / f"globcover_{number:03d}.tif"
        # Write out new land cover
        modified_landcover.rio.to_raster(
            output_path,
            compress="LZW",
            tiled=True,
            BIGTIFF="IF_SAFER"
        )

        return output_path


class ElevationSolution():
    """This class is to change the elevation"""

    def __init__(
            self,
            hydro_combination_path: Path,
            flood_model: str,
            vectors: str = None
    ) -> None:
        """
        Change the elevation based on the vector.
        This class relates to functions:
        - flood_model_executor in lisflood_simulations_generator.py
        - par_generator in lisflood_parameters_generator.py
        - hydrological_and_hydrodynamic_simulation_generator in hydrological_and_hydrodynamic_pipeline.py

        Parameters
        ----------
        hydro_combination_path : Path
            Directory to folder storing all necessary data
        flood_model : str
            Either "lisflood-fp" or "bg-flood"
        vectors : str = None
            Name of dataframe that contains 'vector_path', 'value', 'distance' columns:
            - 'vector_path': Column that stores directories to specific vectors
            - 'value: Column that stores value of the vectors used to increase/decrease elevation
            - 'distance': Column that stores value to smooth the decreased elevation
        """
        self.hydro_combination_path = hydro_combination_path
        self.vectors = pd.read_csv(self.hydro_combination_path / vectors)
        self.flood_model = flood_model

        if flood_model == "lisflood-fp":
            with rxr.open_rasterio(self.hydro_combination_path / "z.asc") as dem:
                self.dem = dem.squeeze().load()

        else:
            terrain_data = xr.open_dataset(self.hydro_combination_path / "8m_geofabric_clipped.nc")
            self.dem = terrain_data.z.squeeze()
            self.roughness_length = terrain_data.zo.squeeze()

    def rasterize_vector(self, vector_path):
        """
        Rasterize vector

        Parameters
        ----------
        vector_path : str
            A directory to vector

        Returns
        -------
        vector_raster : xr.DataArray
            Rasterized vector
        """
        # Read vector
        vector = gpd.read_file(vector_path)

        # Get the shape of vector
        shapes = [(geom, 1) for geom in vector.geometry]

        # Rasterize the vector
        vector_raster = rasterize(
            shapes,
            out_shape=self.dem.shape,
            transform=self.dem.rio.transform(),
            fill=0,
            dtype="uint8"
        )

        return vector_raster

    def increase_elevation(
            self,
            dem: xr.DataArray,
            mask: np.ndarray,
            value: float
    ):
        """
        Increase the elevation

        Parameters
        ----------
        dem : xr.DataArray
            Elevation data read by whitebox tool
        mask : np.ndarray
            Rasterized vector
        value : float
            A value to increase the elevation

        Returns
        -------
        increased_elevation : Raster
            Modified elevation data
        """
        # Create a copy of DEM to modify
        increased_dem = dem.copy()

        # Increase elevation data values
        increased_dem.values[mask == 1] += value

        return increased_dem

    def decrease_elevation(
            self,
            dem: xr.DataArray,
            mask: np.ndarray,
            value: float,
            distance: float = 0
    ):
        """
        Decrease the elevation

        Parameters
        ----------
        dem : Raster
            Elevation data read by whitebox tool
        mask : np.ndarray
            Rasterized vector
        value : float
            Value to decrease the elevation
        distance : float
            Rate to control the sharpness of changing elevation

        Returns
        -------
        decreased_elevation : Raster
            Modified elevation data
        """
        # Create a copy of DEM to modify
        decreased_dem = dem.copy()

        # If no smoothing by distance, sharply change
        if distance <= 0:
            decreased_dem.values[mask == 1] -= abs(value)
            return decreased_dem

        else:
            # Smoothing using distance
            # Here we create a smoothing transition weights.
            # It will mask out area (close to vector) that should be affected (close to 1)
            # and area (far away from the vector) that should not be affected (close to 0)
            dist = distance_transform_edt(mask == 0)
            weight = np.clip(1 - dist / distance, 0, 1)
            decreased_dem.values -= abs(value) * weight

        return decreased_dem

    def change_elevation(self):
        """Change the elevation"""

        # Copy DEM to work separately
        modified_dem = self.dem

        # Change elevation based on each vector
        for _, row in self.vectors.iterrows():

            # Extract information for each vector
            vector_path = row["vector_path"]
            value = row["value"]
            distance = row["distance"]

            # Rasterize vector
            rasterized_vector = self.rasterize_vector(vector_path)

            # Increase elevation
            if value > 0:
                modified_dem = self.increase_elevation(
                    modified_dem,
                    rasterized_vector,
                    value
                )

            # Decrease elevation
            else:
                modified_dem = self.decrease_elevation(
                    modified_dem,
                    rasterized_vector,
                    value,
                    distance
                )

        return modified_dem

    def apply_elevation_solution(self):
        """Apply solution to elevation data"""
        # Change elevation data
        modified_dem = self.change_elevation()

        if self.flood_model == "lisflood-fp":
            # Find existing elevation files
            existing_z_files = sorted(
                self.hydro_combination_path.glob("z_elevation_*.asc")
            )

            # Decide the number of output file
            # based on the number of running the model
            number = len(existing_z_files) + 1

            # Create filename
            output_name = f"z_elevation_{number:03d}.asc"

            # Write out
            modified_dem.rio.to_raster(
                self.hydro_combination_path / output_name,
                compress="LZW",
                tiled=True
            )

        else:
            # Find existing elevation files
            existing_z_files = sorted(
                self.hydro_combination_path.glob("8m_geofabric_clipped_elevation_*.nc")
            )

            # Decide the number of output file
            # based on the number of running the model
            number = len(existing_z_files) + 1

            # Create filename
            output_name = f"8m_geofabric_clipped_elevation_{number:03d}.nc"

            # Merge z and zo
            terrain_data = xr.Dataset({
                "z": modified_dem,
                "zo": self.roughness_length
            })

            # Add CRS
            terrain_data.rio.write_crs("EPSG:2193", inplace=True)

            # Write out
            terrain_data.to_netcdf(self.hydro_combination_path / output_name)