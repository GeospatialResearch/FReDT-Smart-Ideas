# Copyright © 2021-2025 Geospatial Research Institute Toi Hangarau
# LICENSE: https://github.com/GeospatialResearch/Digital-Twins/blob/master/LICENSE
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from osgeo import gdal # Import gdal before rasterio to get rid of DLL error
import subprocess
import rioxarray as rxr

from pathlib import Path

def value_change(
        shapefile_path: Path,
        file_need_changing: str,
        value: float,
        inside: bool = True
) -> None:
    """
    Change pixel values inside or outside polygons

    Parameters
    ----------
    shapefile_path: Path
        Common path to shapefile that cover the area that needs changing
    file_need_changing: str
        Name and path of changed file
    value: float
        Replaced value
    inside: bool = True
        If True, change values inside, else, change values outside.
        Default is True
    """
    # Set up value changing command
    if inside:
        # Change values inside polygons
        subprocess.run([
            'gdal_rasterize',
            '-burn', f'{value}',
            shapefile_path,
            file_need_changing
        ], check=True)

    else:
        # Change values outside polygons
        subprocess.run([
            'gdal_rasterize',
            '-i',
            '-burn', f'{value}',
            shapefile_path,
            file_need_changing
        ], check=True)


class TerrainFilter:
    """This class is to filter terrain data for wflow"""

    def __init__(
            self,
            path: Path,
            hydromt_path: Path,
            river_name: str,
            origin_crs: int = 2193,
            roughness: bool = True,
            origin_filename: str = '8m_geofabric'
    ) -> None:
        """
        Filter terrain data for wflow

        Parameters
        -----------
        path: str
            Common path to the directory that contains necessary files to filter terrain data
        hydromt_path : Path
            A directory to where all necessary files are stored to run wflow model
        river_name: str
            Name of directory to where the river information files are stored
        roughness : bool
            Whether to print out roughness and DEM. Default is True
        origin_filename : str = '8m_geofabric'
            Name of terrain raster filename.
            At the moment, only two names - 8m_geofabric and 4m_geofabric
        """
        self.path = path
        self.hydromt_path = hydromt_path
        self.river_name = river_name
        self.origin_crs = origin_crs
        self.roughness = roughness
        self.origin_filename = origin_filename

    def terrain_splitting(self) -> None:
        """Split terrain data into DEM and roughness (if any)"""
        # Get parental directory to terrain files
        terrain_parent_files = self.hydromt_path / f"river_data/{self.river_name}"

        # Get list of terrain files
        terrain_file_path = list(terrain_parent_files.glob(f"{self.origin_filename}.nc"))
        terrain = rxr.open_rasterio(terrain_file_path[0])

        # Save as tif
        if self.roughness:
            terrain['z'].rio.to_raster(self.path / f"{self.origin_filename}_dem_split.tif")  # Name here is constant
            terrain['zo'].rio.to_raster(self.path / f"{self.origin_filename}_roughness_split.tif")  # Name here is constant
        else:
            terrain.rio.to_raster(self.path / f"{self.origin_filename}_dem_split.tif")

    def remove_sea(self) -> None:
        """Clip sea area mainly in DEM and roughness"""
        # Get New Zealand shapefile
        nz_shapefile = self.hydromt_path / "nz_coastline.shp"

        # Files need changing
        dem = self.path / f"{self.origin_filename}_dem_split.tif"  # Name here is constant
        roughness = self.path / f"{self.origin_filename}_roughness_split.tif"  # Name here is constant

        # Remove sea by changing sea area into nodata value (-9999)
        value_change(nz_shapefile, dem, -9999, False)
        value_change(nz_shapefile, roughness, -9999, False)

    def nodata_filling(self) -> None:
        """Fill nodata value with -9999"""
        # Fill the nodata value
        dem_nosea = rxr.open_rasterio(self.path / f"{self.origin_filename}_dem_split.tif",
                                      chunks={"x": 4096, "y": 4096})  # Name here is constant
        dem_replace_nodata = dem_nosea.fillna(-9999)
        dem_write_nodata = dem_replace_nodata.rio.write_nodata(-9999)
        dem_write_nodata.rio.to_raster(self.path / f"{self.origin_filename}_dem_for_wflow.tif")  # Name here is constant

        roughness_nosea = rxr.open_rasterio(self.path / f"{self.origin_filename}_roughness_split.tif")  # Name here is constant
        roughness_replace_nodata = roughness_nosea.fillna(-9999)
        roughness_write_nodata = roughness_replace_nodata.rio.write_nodata(-9999)
        roughness_write_nodata.rio.to_raster(
            self.path / f"{self.origin_filename}_roughness_for_wflow.tif", # Name here is contstant
            tiled=True,
            windowed=True,
            lock=False,
        )

    def filter_dem_for_wflow(self) -> None:
        """Convert DEM into version that can be used by wflow"""
        # Split terrain data into DEM and roughness
        self.terrain_splitting()

        # Remove sea
        self.remove_sea()

        # Fill nodata
        self.nodata_filling()
