# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:28:16 2026

@author: mng42
"""
import logging
from datetime import datetime
from pathlib import Path

from osgeo import gdal  # Import gdal before rasterio

import netCDF4
import numpy as np
import xarray as xr
from rasterio.enums import Resampling
from tqdm import tqdm

from eddie.digitaltwin.utils import setup_logging, LogLevel

setup_logging(LogLevel.DEBUG)
log = logging.getLogger(__name__)


class PrecipitationGenerator():
    """This class is to generate precipitation"""

    def __init__(
            self,
            flood_model_path: Path,
            precipitation_path: Path,
            terrain_bounding_box: xr.Dataset,
            start_time: datetime,
            end_time: datetime,
            crs: int = 2193
    ) -> None:
        """
        Generate precipitation

        Parameters
        ----------
        flood_model_path : Path
            Directory to folder storing flood model data
        precipitation_path : Path
            Directory to folder storing precipitation data
        terrain_bounding_box: Polygon
            Bounding's box of terrain data
        start_time : datetime
            Starting time details. Format is "yyyy-mm-ddThh:mm:ss"
        end_time : datetime
            Ending time details.
        crs : int, optional
            Targeted crs. The default is 2193 for NZTM.
        """
        self.flood_model_path = flood_model_path
        self.precipitation_path = precipitation_path
        self.terrain_bounding_box = terrain_bounding_box
        self.start_time = start_time
        self.end_time = end_time
        self.crs = crs

    def extract_precipitation_within_time(self) -> xr.Dataset:
        """
        Extract precipitation within given time

        Returns
        -------
        precipitation_subset : xr.Dataset
            Precipitation data within given time
        """
        log.info("Extracting precipitation data")

        # Extract time information (month and year) from start_time.
        # This is just within a month.
        # But it should be able to read different months,
        # and it will be scripted in the future
        given_year = self.start_time.year
        given_month = f"{self.start_time.month:02d}"  # Ex: 01, 02, etc.

        # Set up path to precipitation file
        precipitation_name = f"precipitation_nz_{given_year}{given_month}.nc"
        precipitation_given_time_path = self.precipitation_path / precipitation_name

        # Read the precipitation file that contains given time
        precipitation_given_time = xr.open_dataset(precipitation_given_time_path)

        # Extract precipitation within time
        precipitation_subset = precipitation_given_time.sel(
            time=slice(self.start_time, self.end_time)
        )

        return precipitation_subset

    def format_precipitation(
            self,
            precipitation_subset: xr.Dataset
    ) -> xr.Dataset:
        """
        Format precipitation for easy processing, this includes:
            - rename variable
            - set spatial dimensions
            - Reproject crs

        Parameters
        ----------
        precipitation_subset : xr.Dataset
            Precipitation data within given time

        Returns
        -------
        formatted_precipitation_subset : xr.Dataset
            Formatted precipitation for easy processing within given time
        """
        log.info("Formatting precipitation data")
        # Rename variable
        precipitation_subset = precipitation_subset.rename({
            'pr': 'rainfall_depth'
        })

        # Set spatial dimensions
        precipitation_subset = precipitation_subset.rio.set_spatial_dims(
            x_dim='lon',
            y_dim='lat'
        )

        # Set crs to make sure the precipitation has crs
        precipitation_subset = precipitation_subset.rio.write_crs("EPSG:4326")

        # Reproject crs
        formatted_precipitation_subset = precipitation_subset.rio.reproject("EPSG:2193")

        return formatted_precipitation_subset

    def padding_box_generator(
            self,
            padding_value: int
    ) -> list[dict]:
        """
        Generate padding box to clip precipitation data

        Parameters
        ----------
        padding_values: int
            Value of padding

        Returns
        -------
        terrain_padding_box : list[dict]
            Padding box of terrain data.
            This is wider than bounding box.
        """
        xmin, ymin, xmax, ymax = self.terrain_bounding_box.bounds

        padding_box = [{
            'type': 'Polygon',
            'coordinates': [[
                [xmin - padding_value, ymin - padding_value],
                [xmin - padding_value, ymax + padding_value],
                [xmax + padding_value, ymax + padding_value],
                [xmax + padding_value, ymin - padding_value],
                [xmin - padding_value, ymin - padding_value]
            ]]
        }]

        return padding_box

    def clip_precipitation(
            self,
            padding_value: int,
            precipitation_data: xr.Dataset
    ) -> xr.Dataset:
        """
        Clip precipitation data, the steps include:
            - clip precipitation data
            - convert clipped values to float32
            - empty attributes of clipped precipitation

        Parameters
        ----------
        padding_value : int
            Value of padding
        precipitation_data : xr.Dataset
            Precipitation data that needs clipping

        Returns
        -------
        clipped_precipitation : xr.Dataset
            Clipped precipitation data
        """
        log.info("Clipping precipitation data")
        # Generate padding box
        precipitation_padding_box = self.padding_box_generator(padding_value)

        # Clip precipitation
        clipped_precipitation = precipitation_data.rio.clip(
            precipitation_padding_box, from_disk=True
        )

        # Convert clipped values to float32 to easily procress in the future
        clipped_precipitation = clipped_precipitation.astype('float32')

        # Empty attributes of precipitation data after clipping
        clipped_precipitation.attrs = {}

        return clipped_precipitation

    def reproject_precipitation(
            self,
            precipitation_data: xr.Dataset
    ) -> xr.Dataset:
        """
        Reproject precipitation data

        Parameters
        ----------
        precipitation_data : xr.Dataset
            Precipitation data

        Returns
        -------
        reprojected_precipitation_data : xr.Dataset
            Reprojected precipitation data
        """
        log.info("Reprojecting precipitation data")
        # Reproject precipitation data to padding box of terrain data
        reprojected_precipitation_data = precipitation_data.rio.reproject(
            dst_crs='EPSG:2193',
            resolution=8,  # 8m for now. It will be changed into coded resolution in the future
            resampling=Resampling.nearest
        )

        return reprojected_precipitation_data

    def write_out_precipitation(
            self,
            precipitation_path: Path,
            precipitation_data: xr.Dataset
    ):
        """
        Write out precipitation as netCDF file

        Parameters
        ----------
        precipitation_path : Path
            A directory where precipitation data is stored
        precipitation_data : xr.Dataset
            Precipitation data that needs writing out
        """
        # Write out precipitation data as netCDF file
        precipitation_data.to_netcdf(
            precipitation_path,
            engine='netcdf4'
        )

    def format_each_precipitation_timestep(
            self,
            clipped_precipitation
    ):
        """
        Format each precipitation timestep from clipped precipitation data

        Parameters
        ----------
        clipped_precipitation : TYPE
            DESCRIPTION.
        """
        log.info("Writing out precipitation data")
        # Create fine precipitation folder
        fine_precipitation_folder = self.flood_model_path / "precipitation"
        fine_precipitation_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        # Loop through each precipitation time step to format and write out
        for i, t in tqdm(list(enumerate(clipped_precipitation.time)), desc="Formatting precipitation"):
            # Extract each precipitation timestep
            each_precipitation_timestep = clipped_precipitation.sel(time=t)

            # Reproject each precipitation timestep
            reprojected_each_precipitation_timestep = self.reproject_precipitation(each_precipitation_timestep)

            # Clip each precipitation timestep
            clipped_reprojected_each_precipitation_timestep = self.clip_precipitation(
                1160,  # This value is constant
                reprojected_each_precipitation_timestep
            )

            # Write out to precipitation folder
            fine_precipitation_path = fine_precipitation_folder / f"precipitation_{i:03d}.nc"
            self.write_out_precipitation(
                fine_precipitation_path,
                clipped_reprojected_each_precipitation_timestep
            )

    def collect_precipitation_timesteps(self) -> list:
        """
        Collect all precipitation timesteps' files

        Returns
        -------
        precipitation_timesteps_files : list
            List of all files of precipitation timesteps
        """
        # Collect files of all precipitation timesteps
        precipitation_timesteps_files = sorted(
            self.flood_model_path.glob("precipitation/precipitation_*.nc")
        )

        return precipitation_timesteps_files

    def combine_precipitation_timesteps(self) -> xr.Dataset:
        """
        Read and write out all precipitation timesteps into one precipitation data

        Returns
        -------
        combined_precipitation_timestep : xr.Dataset
            Precipitation that combines all timesteps
        """
        log.info("Combining precipitation timesteps")
        # Collect all files of precipitation timesteps
        precipitation_timesteps_files = self.collect_precipitation_timesteps()

        # Read/combine all files of precipitation timesteps
        combined_precipitation_timesteps = xr.open_mfdataset(
            precipitation_timesteps_files,
            combine='nested',
            concat_dim='time'
        )

        # Change variable name from rainfall_depth to depth
        combined_precipitation_timesteps = combined_precipitation_timesteps.rename({
            "rainfall_depth": "depth"
        })

        # remove spatial_ref (important)
        if "spatial_ref" in combined_precipitation_timesteps:
            combined_precipitation_timesteps = combined_precipitation_timesteps.drop_vars("spatial_ref")

        # Add necessary attribute
        combined_precipitation_timesteps.attrs["Conventions"] = "CF-1.5"

        return combined_precipitation_timesteps

    def precipitation_data_generator(self) -> xr.Dataset:
        """
        Generate precipitation data for flood model (LISFLOOD-FP)

        Returns
        -------
        combined_precipitation_timesteps : xr.Dataset
            Precipitation that combines all timesteps
        """
        ## Clip coarse precipitation before fine resolution
        # Extract precipitation within given time
        coarse_precipitation_subset = self.extract_precipitation_within_time()

        # Format coarse precipitation
        formatted_coarse_precipitation_subset = self.format_precipitation(coarse_precipitation_subset)

        # Clip coarse precipitation
        clipped_coarse_precipitation = self.clip_precipitation(
            12600,
            formatted_coarse_precipitation_subset
        )

        # Format each precipitation timestep
        self.format_each_precipitation_timestep(clipped_coarse_precipitation)

        # Generate precipitation data that combines all timesteps
        combined_precipitation_timesteps = self.combine_precipitation_timesteps()

        return combined_precipitation_timesteps


class PrecipitationFloodModelGenerator():
    """This class is to generate precipitation for flood model"""

    def __init__(
            self,
            flood_model_path: Path,
            combined_precipitation_data: xr.Dataset
    ) -> None:
        """
        Generate precipitaiton for flood model (LISFLOOD-FP).
        The outfile_precipitation variable will be used throughout this class

        Parameters
        ----------
        flood_model_path : Path
            Directory to folder storing flood model data
        combined_precipitation_data : xr.Dataset
            Precipitation data where all timesteps are combined and processed
        """
        self.flood_model_path = flood_model_path
        self.combined_precipitation_data = combined_precipitation_data

    def assign_each_precipitation_timestep(
            self,
            precipitation_var: netCDF4.Variable
    ) -> None:
        """
        Assign precipitation values to precipitation variable

        Parameters
        -------
        precipitation_var : netCDF4.Variable
            Precipitation variable that needs assigning values separately
        """

        for i, t in enumerate(self.combined_precipitation_data.time):
            # Extract each precipitation timesteps
            each_precipitation_timestep = self.combined_precipitation_data.sel(time=t)['depth']

            # Convert units to mm/hr
            each_precipitation_timestep = each_precipitation_timestep.astype('float32') * 3600

            # Write to precipitation variable
            precipitation_var[i, :, :] = each_precipitation_timestep.values

    def precipitation_generator(self) -> netCDF4.Variable:
        """
        Generate variables of outfile precipitation

        Parameters
        ----------
        outfile_precipitation : netCDF4.Dataset
            Precipitation data written out as netCDF file with dimensions
        outfile_precipitation_time : int
            Time variable of outfile precipitation

        Returns
        -------
        precipitation_var : netCDF4.Variable
            Precipitation variable that needs assigning values separately
        """
        # Open netCDF precipitation file
        outfile_precipitation_path = self.flood_model_path / "precipitation_dynamic.nc"
        outfile_precipitation = netCDF4.Dataset(
            outfile_precipitation_path,
            'w', format='NETCDF4'
        )

        # Extract dimensions from the precipitation dat
        outfile_precipitation_x = self.combined_precipitation_data.x.size
        outfile_precipitation_y = self.combined_precipitation_data.y.size
        outfile_precipitation_time = self.combined_precipitation_data.time.size

        # Generate dimensions in precipitation netCDF file
        outfile_precipitation.createDimension(
            'time',
            outfile_precipitation_time
        )
        outfile_precipitation.createDimension(
            'x',
            outfile_precipitation_x
        )
        outfile_precipitation.createDimension(
            'y',
            outfile_precipitation_y
        )

        # Create variables
        time_var = outfile_precipitation.createVariable(
            'time', 'float32', ('time',)
        )
        x_var = outfile_precipitation.createVariable(
            'x', 'float64', ('x',)
        )
        y_var = outfile_precipitation.createVariable(
            'y', 'float64', ('y',)
        )
        precipitation_var = outfile_precipitation.createVariable(
            'depth', 'float32', ('time', 'y', 'x')
        )

        # Add units
        time_var.units = 'second'
        x_var.units = 'm'
        y_var.units = 'm'
        precipitation_var.units = 'kg m-2'
        time_var.axis = 'T'
        x_var.axis = 'X'
        y_var.axis = 'Y'

        # Assign values
        time_var[:] = np.arange(outfile_precipitation_time) * 3600
        x_var[:] = self.combined_precipitation_data.x.values
        y_var[:] = self.combined_precipitation_data.y.values

        # Assign each precipitation timestep
        self.assign_each_precipitation_timestep(precipitation_var)

        # Close writing-out file
        outfile_precipitation.close()

    def precipitation_for_flood_model_generator(self) -> None:
        """Generate precipitation for flood model (LISFLOOD-FP)"""

        # Generate precipitation for flood model
        self.precipitation_generator()


