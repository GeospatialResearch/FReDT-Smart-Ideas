from osgeo import gdal
import rioxarray as rxr
import xarray as xr
from pathlib import Path
import geopandas as gpd
import pandas as pd
from rasterio.features import rasterize

from whitebox_workflows import WbEnvironment, Raster
from whitebox.whitebox_tools import WhiteboxTools

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
            polygons: str = None
    ) -> None:
        """
        Change the land cover based on polygons

        Parameters
        ----------
        hydro_combination_path : Path
            Directory to folder storing all necessary data
        hydromt_path : Path
            A directory to where all necessary files are stored to run wflow model
        polygons : str = None
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


    def apply_landcover_solution(self):
        """This is to change the landcover based on polygons"""
        # Read current land cover data
        with rxr.open_rasterio(self.hydromt_path / r'globcover_origin.tif') as current_landcover:
            current_landcover = current_landcover.squeeze().load()

        # Read polygons
        polygons_path = self.hydro_combination_path / self.polygons
        polygons = gpd.read_file(polygons_path)

        # Convert crs
        # This step will be removed in future
        polygons_crs = polygons.to_crs(4326)

        # Rasterize and apply new values to current land cover
        modified_landcover = self.rasterize_polygons(
            current_landcover,
            polygons_crs
        )

        # Write out new land cover
        modified_landcover.rio.to_raster(
            self.hydromt_path / "globcover.tif",
            compress="LZW",
            tiled=True,
            BIGTIFF="IF_SAFER"
        )


class ElevationSolution():
    """This class is to change the elevation"""

    def __init__(
            self,
            hydro_combination_path: Path,
            vectors: str = None
    ) -> None:
        """
        Change the elevation based on the vector

        Parameters
        ----------
        hydro_combination_path : Path
            Directory to folder storing all necessary data
        vectors : str = None
            Name of dataframe that contains 'vector_path', 'value', 'distance' columns:
            - 'vector_path': Column that stores directories to specific vectors
            - 'value: Column that stores value of the vectors used to increase/decrease elevation
            - 'distance': Column that stores value to smooth the decreased elevation
        """
        self.hydro_combination_path = hydro_combination_path
        self.vectors = pd.read_csv(self.hydro_combination_path / vectors)

        self.dem = wbe.read_raster(str(self.hydro_combination_path / r"z.asc"))

    def increase_elevation(
            self,
            dem: Raster,
            vector_path: Path,
            value: float
    ):
        """
        Increase the elevation

        Parameters
        ----------
        dem : wbe.Raster
            Elevation data read by whitebox tool
        vector_path : Path
            A directory to a specific vector
        value : int
            A value to increase the elevation

        Returns
        -------
        increased_elevation : Raster
            Modified elevation data
        """
        # Raise elevation data values
        increased_elevation = wbe.raise_walls(
            dem,
            wbe.read_vector(str(vector_path)),
            wall_height=value
        )

        return increased_elevation

    def decrease_elevation(
            self,
            dem: Raster,
            vector_path: Path,
            value: float,
            distance: float
    ):
        """
        Decrease the elevation

        Parameters
        ----------
        dem : Raster
            Elevation data read by whitebox tool
        vector_path : Path
            A directory to a specific vector
        value : float
            Value to decrease the elevation
        distance : float
            Rate to control the sharpness of changing elevation

        Returns
        -------
        decreased_elevation : Raster
            Modified elevation data
        """
        # Decrease elevation data
        decreased_elevation = wbe.burn_streams(
            dem,
            wbe.read_vector(str(vector_path)),
            decrement_value=abs(value),
            gradient_distance=distance
        )

        return decreased_elevation

    def change_elevation(self):
        """Change the elevation"""

        # Copy DEM to work separately
        modified_dem = self.dem

        # Change elevation based on each vector
        for idx, row in self.vectors.iterrows():

            vector_path = row["vector_path"]
            value = row["value"]
            distance = row["distance"]

            # Increase elevation
            if value > 0:
                modified_dem = self.increase_elevation(
                    modified_dem,
                    vector_path,
                    value
                )

            # Decrease elevation
            else:
                modified_dem = self.decrease_elevation(
                    modified_dem,
                    vector_path,
                    value,
                    distance
                )

        return modified_dem

    def apply_elevation_solution(self):
        """Apply solution to elevation data"""
        # Change elevation data
        modified_dem = self.change_elevation()

        # Write out
        wbe.write_raster(
            modified_dem,
            str(self.hydro_combination_path / "z.asc"),
            compress=False
        )

