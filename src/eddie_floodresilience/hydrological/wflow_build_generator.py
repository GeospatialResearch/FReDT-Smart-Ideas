# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 09:01:33 2026

@author: mng42
"""

import yaml
from pathlib import Path
from datetime import datetime


class WflowBuildGenerator():
    """This class is to generate wflow_build.yml for preprocessing data for wflow"""
    
    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        resolution: float,
        wflow_model_path: Path,
        forcing_path: Path
    ) -> None:
        """
        Generate wflow_build.yml for preprocessing data for wflow.
        This wflow_build.yml matches with information (mostly directory information)
        from data_catalog.yml.

        Parameters
        ----------
        start_time : datetime
            Starting time of simulation.
            This should include the spin-up time. 
            Normally, it is 1-year before the flood event.
        end_time : datetime
            Ending time of simulation
            This should include some periods of time after the flood event.
            Normally, it is about 12 hours or 1 day.
        resolution : float
            Resolution for flow data. 
            Default is 0.00045 (in crs 4326) ~ 50 m (in crs 2193)
        wflow_model_path: Path
            A directory to where the data_catalog.yml is stored and to run wflow model
        forcing_path: Path
            A directory to where the forcing files are stored
        """
        self.start_time = start_time.replace(year=start_time.year - 1)
        self.end_time = end_time
        self.resolution = resolution
        self.wflow_model_path = wflow_model_path
        self.forcing_path = forcing_path

    def config_section(self) -> dict:
        """
        Write out configuration section
        
        Returns
        -------
        config : dict
            A dictionary that contains configuration section
        """
        # Set up path for forcing
        if str(self.forcing_path).endswith(".nc"):
            input_path_forcing = str(self.forcing_path)
        else:
            input_path_forcing = "era5_hourly_new.nc"

        # Generate configuration section
        config = {
            "setup_config": {
                "starttime": self.start_time,
                "endtime": self.end_time,
                "timestepsecs": 3600,
                "input.path_forcing": input_path_forcing,
                # Extra parameters
                "water_mass_balance__flag": True,
                "output.path": "output.nc",
                "output.compressionlevel": 1,
                "netcdf.path": "output_scalar.nc",
                "output.vertical.actevap": "actevap",
                "output.vertical.satwaterdepth": "satwaterdepth",
                "output.vertical.ustoredepth": "ustoredepth",
                "output.vertical.total_storage": "total_storage",
                "output.lateral.land.h": "h_land",
                "output.lateral.land.h_av": "h_av_land",
                "output.lateral.river.h": "h_river",
                "output.lateral.river.h_av": "h_av_river",
                "output.lateral.river.q": "q_river",
                "output.lateral.river.q_av": "q_av_river"
            }
        }
        
        return config
    
    def basemaps_section(self) -> dict:
        """
        Write out basemaps' section
        
        Returns
        -------
        basemaps : dict
            A dictionary that contains basemaps' section
        """
        # Generate basemaps section
        basemaps = {
            "setup_basemaps": {
                "hydrography_fn": "merit_hydrox",
                "basin_index_fn": "merit_hydro_index",
                "upscale_method": "ihu",
                "res": self.resolution
            }
        }
        
        return basemaps
    
    def rivers_section(self) -> dict:
        """
        Write out rivers' section

        Returns
        -------
        rivers : dict
            A dictionary that contains rivers' section
        """
        # Generate rivers section
        rivers = {
            "setup_rivers": {
                "hydrography_fn": "merit_hydrox",
                "river_geom_fn": "hydro_rivers_lin",
                "river_upa": 0.1,
                "rivdph_method": "manning",
                "min_rivdph": 1,
                "min_rivwth": 0.05,
                "slope_len": 2000,
                "smooth_len": 5000,
                "river_routing": "kinematic-wave"
            }
        }
        
        return rivers
    
    def lakes_section(self) -> dict:
        """
        Write out lakes' section

        Returns
        -------
        lakes : dict
            A dictionary that contains lakes' section
        """
        # Generate lakes section
        lakes = {
            "setup_lakes": {
                "lakes_fn": "hydro_lakes",
                "min_area": 10.0
            }
        }
        
        return lakes
    
    def landcover_section(self) -> dict:
        """
        Write out landcover's section
        
        Returns
        -------
        lulc : dict
            A dictionary that contains landcover's section
        """
        # Generate landuse/landcover's section
        landcover = {
            "setup_lulcmaps": {
                "lulc_fn": "globcover_2009",
                "lulc_mapping_fn": "globcover_mapping_default"
            }
        }
        
        return landcover
    
    def lai_section(self) -> dict:
        """
        Write out LAI section

        Returns
        -------
        lai : dict
            A dictionary that contains lai's section
        """
        # Generate lai section
        lai = {
            "setup_laimaps": {
                "lai_fn": "modis_lai",
                "lulc_fn": "globcover_2009",
                "lulc_sampling_method": "any",
                "lulc_zero_classes": [200, 210, 220],
                "buffer": 2
            }    
        }
        
        return lai
    
    def soil_section(self) -> dict:
        """
        Write out soil section
        
        Returns
        -------
        soil : dict
            A dictionary that contains soil's section
        """
        # Generate soil section
        soil = {
            "setup_soilmaps": {
                "soil_fn": "soilgrids_2020",
                "ptf_ksatver": "brakensiek"
            }
        }
        
        return soil
    
    def gauges_section(self) -> dict:
        """
        Write out gauges' section
        
        Returns
        -------
        gauges : dict
            A dictionary that contains gauges' section
        """
        # Generate gauges' section
        gauges = {
            "setup_gauges": {
                "gauges_fn": "grdc",
                "snap_to_river": True,
                "derive_subcatch": False
            }
        }
        
        return gauges

    def precipitation_section(self) -> dict:
        """
        Write out precipitation's section

        Returns
        -------
        precipitation : dict
            A dictionary that contains precipitation's section
        """
        # Generate precipitation's section
        precipitation = {
            "setup_precip_forcing": {
                "precip_fn": "era5_hourly",
                "chunksize": 48
            }
        }

        return precipitation

    def temperature_section(self) -> dict:
        """
        Write out temperature section

        Returns
        -------
        temp_pet : dict
            A dictionary that contains temperature section
        """
        # Generate temperature section
        temperature = {
            "setup_temp_pet_forcing": {
                "temp_pet_fn": "era5_hourly",
                "press_correction": True,
                "temp_correction": True,
                "dem_forcing_fn": "era5_orography",
                "skip_pet": True,
                "chunksize": 48
            }
        }

        return temperature

    def potential_evaporation_section(self) -> dict:
        """
        Write out potential evaporation

        Returns
        -------
        potential_evaporation : dict
            A dictionary that contains potential evaporation section
        """
        # Generate potential evaporation section
        potential_evaporation = {
            "setup_pet_forcing": {
                "pet_fn": "era5_hourly",
                "chunksize": 48
            }
        }

        return potential_evaporation

    def constant_parameters_section(self) -> dict:
        """
        Write out constant parameters' section
        
        Returns
        -------
        constant_parameters : dict
            A dictionary that contains constant parameters' section
        """
        # Generate constant parameters
        constant_parameters = {
            "setup_constant_pars": {
                "KsatHorFrac": 1,
                "Cfmax": 3.75653,
                "cf_soil": 0.038,
                "EoverR": 0.11,
                "InfiltCapPath": 5,
                "InfiltCapSoil": 1,
                "MaxLeakage": 0,
                "rootdistpar": -500,
                "TT": 0,
                "TTI": 2,
                "TTM": 0,
                "WHC": 0.1,
                "G_Cfmax": 5.3,
                "G_SIfrac": 0.002,
                "G_TT": 1.3,
                "f": 20
            }
        }
        
        return constant_parameters
        
    def write_section(self) -> dict:
        """
        Write out "write" section.
        This section is to provide conditions for some files to be written correctly
        
        Returns
        -------
        write : dict
            A dictionary that contains conditions for some files to be written
        """
        # Generate "write" section
        write_section = {
            "write_forcing": {"freq_out": "YE"},
            "write_grid": {},
            "write_geoms": {},
            "write_config": {}
        }
        
        return write_section
    
    def wflow_build_section(self) -> dict:
        """
        Organise wflow build's section

        Returns
        -------
        wflow_build : dict
            A dictionary that contains wflow build's section
        """
        # Set up wflow build dictionary
        wflow_build = {}
        
        # Set up sections list
        if str(self.forcing_path).endswith(".nc"):
            sections_list = [
                self.config_section(),
                self.basemaps_section(),
                self.rivers_section(),
                self.lakes_section(),
                self.landcover_section(),
                self.lai_section(),
                self.soil_section(),
                self.gauges_section(),
                self.constant_parameters_section(),
                self.write_section()
            ]
        else:
            sections_list = [
                self.config_section(),
                self.basemaps_section(),
                self.rivers_section(),
                self.lakes_section(),
                self.landcover_section(),
                self.lai_section(),
                self.soil_section(),
                self.gauges_section(),
                self.precipitation_section(),
                self.temperature_section(),
                self.potential_evaporation_section(),
                self.constant_parameters_section(),
                self.write_section()
            ]
        
        # Generate wflow build section
        for each_section in sections_list:
            wflow_build.update(each_section)
        
        return wflow_build
            
    def write_out_wflow_build(
            self,
            wflow_build
        ) -> None:
        """
        Write out wflow_build.yml
        
        Parameters
        ----------
        wflow_build : dict
            A dictionary contains information of all sections
        """
        # Set up output filename
        output_filename = self.wflow_model_path / "wflow_build.yml"
        
        # Geenrate content for wflow_build.yml
        with open(output_filename, "w") as output_file:
            yaml.dump(
                wflow_build,
                output_file,
                sort_keys=False
            )
    
    def wflow_build_generator(self):
        """Generate data_catalog.yml file"""
        # Set up content for wflow_build file
        wflow_build = self.wflow_build_section()
        
        # Write wflow_build file
        self.write_out_wflow_build(wflow_build)
    