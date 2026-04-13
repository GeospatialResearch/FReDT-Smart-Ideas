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
            origin_crs: int = 2193,
            converted_crs: int = 4326,
            roughness: bool = True
    ) -> None:
        """
        Filter terrain data for wflow

        Parameters
        -----------
        path: str
            Common path to the directory that contains necessary files to filter terrain data
        origin_crs : int = 2193
            Original CRS (default is 2193)
        converted_crs : int = 4326
            Converted CRS (default is 4326)
        roughness : bool
            Whether to print out roughness and DEM. Default is True
        """
        self.path = path
        self.origin_crs = origin_crs
        self.converted_crs = converted_crs
        self.roughness = roughness

    def dem_crs_conversion(self) -> None:
        """Convert DEM CRS. Here default is to convert DEM CRS from 2193 to 4326"""
        # Get DEM
        dem = rxr.open_rasterio(self.path / "8m_geofabric.nc")

        # Ensure the origin CRS is attached
        dem_origin_crs = dem.rio.write_crs(f"EPSG:{self.origin_crs}", inplace=True)

        # Reproject the crs to converted crs
        dem_converted_crs = dem_origin_crs.rio.reproject(f"EPSG:{self.converted_crs}")

        # Save as tif
        if self.roughness:
            dem_converted_crs['z'].rio.to_raster(self.path / "dem_converted_crs.tif")
            dem_converted_crs['zo'].rio.to_raster(self.path / "roughness_converted_crs.tif")
        else:
            dem_converted_crs.rio.to_raster(self.path / "dem_converted_crs.tif")

    def remove_sea(self) -> None:
        """Clip sea area mainly in DEM and roughness"""
        # Get New Zealand shapefile
        nz_shapefile = self.path / "nz_coastline_4326.shp"

        # Files need changing
        dem = self.path / "dem_converted_crs.tif"
        roughness = self.path / "roughness_converted_crs.tif"

        # Remove sea by changing sea area into nodata value (-9999)
        value_change(nz_shapefile, dem, -9999, False)
        value_change(nz_shapefile, roughness, -9999, False)

    def nodata_filling(self) -> None:
        """Fill nodata value with -9999"""
        # Fill the nodata value
        dem_nosea = rxr.open_rasterio(self.path / "dem_converted_crs.tif")
        dem_replace_nodata = dem_nosea.fillna(-9999)
        dem_write_nodata = dem_replace_nodata.rio.write_nodata(-9999)
        dem_write_nodata.rio.to_raster(self.path / "dem_for_wflow.tif")

        roughness_nosea = rxr.open_rasterio(self.path / "roughness_converted_crs.tif")
        roughness_replace_nodata = roughness_nosea.fillna(-9999)
        roughness_write_nodata = roughness_replace_nodata.rio.write_nodata(-9999)
        roughness_write_nodata.rio.to_raster(self.path / "roughness_for_wflow.tif")

    def filter_dem_for_wflow(self) -> None:
        """Convert DEM into version that can be used by wflow"""
        # Convert DEM and roughness CRS (default: 2193 --> 4326)
        self.dem_crs_conversion()

        # Remove sea
        self.remove_sea()

        # Fill nodata
        self.nodata_filling()
