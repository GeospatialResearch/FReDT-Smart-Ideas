# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 09:01:33 2026

@author: mng42
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from dateutil.relativedelta import relativedelta
import geopandas as gpd
import yaml

from src.eddie_floodresilience.solutions.nature.landcover import LandcoverClassDataset

log = logging.getLogger(__name__)


class _IndentedDumper(yaml.SafeDumper):
    """Indent list items under their parent key, keep short lists inline"""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def _represent_list(dumper, data):
    inline = all(not isinstance(item, (dict, list)) for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=inline)


_IndentedDumper.add_representer(list, _represent_list)


class WflowBuildGenerator:
    """
    This class is to generate wflow_build.yml for preprocessing data for wflow

    Attributes
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
    hydromt_path : Path
        A directory to where all necessary files are stored to run wflow model
    river_name : str
        Name of directory to where the river information files are stored
    forcing_path : Path
        A directory to where the forcing files are stored
    scenario_and_id_folder : Path
        Directory to the scenario folder name with ID
    polygons : gpd.GeoDataFrame | None = None
        This polygon dataframe has 'landcover' column with new values
    landcover : LandcoverClassDataset = LandcoverClassDataset.GLOBCOVER = 'globcover'
        Name of land cover dataset. Default is globcover
    """  # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        resolution: float,
        hydromt_path: Path,
        river_name: str,
        forcing_path: Path,
        scenario_and_id_folder: Path,
        polygons: gpd.GeoDataFrame = None,
        landcover: LandcoverClassDataset = LandcoverClassDataset.GLOBCOVER
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
        hydromt_path: Path
            A directory to where all necessary files are stored to run wflow model
        river_name: str
            Name of directory to where the river information files are stored
        forcing_path: Path
            A directory to where the forcing files are stored
        scenario_and_id_folder : Path
            Directory to the scenario folder name with ID
        polygons : gpd.GeoDataFrame | None = None
            This polygon dataframe has 'landcover' column with new values
        landcover : LandcoverClassDataset = LandcoverClassDataset.GLOBCOVER = 'globcover'
            Name of land cover dataset. Default is global cover
        """
        self.start_time = start_time - relativedelta(months=2)
        self.end_time = end_time
        self.resolution = resolution
        self.hydromt_path = hydromt_path
        self.river_name = river_name
        self.forcing_path = forcing_path
        self.scenario_and_id_folder = scenario_and_id_folder
        self.polygons = polygons
        self.landcover = landcover

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

        if self.polygons is not None:
            if str(self.forcing_path).endswith(".nc"):
                # Generate configuration section
                config = {
                    "setup_config": {
                        "data": {
                            "starttime": self.start_time,
                            "endtime": self.end_time,
                            "timestepsecs": 3600,
                            "input.path_forcing": str(self.forcing_path)
                        }
                    }
                }
            else:
                forcing_folder = r"hydrological_process/wflow_test_full/era5_hourly_new_*.nc"
                forcing_path = self.scenario_and_id_folder / forcing_folder

                # Generate configuration section
                config = {
                    "setup_config": {
                        "data": {
                            "starttime": self.start_time,
                            "endtime": self.end_time,
                            "timestepsecs": 3600,
                            "input.path_forcing": str(forcing_path)
                        }
                    }
                }

        else:
            # Generate data section
            data = {
                # Path parameter
                "dir_output": "run_default",

                # Time parameters
                "time.starttime": self.start_time,
                "time.endtime": self.end_time,
                "time.timestepsecs": 3600,

                # Model parameters
                "model.type": "sbm_gwf",
                "model.snow__flag": False,
                "model.snow_gravitational_transport__flag": False,
                "model.river_kinematic_wave__time_step": 900,
                "model.land_kinematic_wave__time_step": 3600,
                "model.land_streamorder__min_count": 2,
                "model.river_streamorder__min_count": 3,
                "model.conductivity_profile": "exponential",
                "water_mass_balance__flag": True,

                # Forcing parameters
                "input.path_forcing": input_path_forcing,
                "input.forcing.atmosphere_water__precipitation_volume_flux": "precip",
                "input.forcing.land_surface_water__potential_evaporation_volume_flux": "pet",
                "input.forcing.atmosphere_air__temperature": "temp",

                # Groundwater parameters
                # Spatial variables
                "input.static.subsurface_surface_water__horizontal_saturated_hydraulic_conductivity": "conductivity",
                "input.static.subsurface_water__specific_yield": "specific_yield",
                "input.static.river_water__infiltration_conductance": "infilt_cond",
                "input.static.river_water__exfiltration_conductance": "exfilt_cond",
                "input.static.river_bottom__elevation": "river_bottom",
                "input.static.subsurface__horizontal_saturated_hydraulic_conductivity_scale_parameter": "gwf_f",
                # Constant variables
                "input.static.soil_layer_water__vertical_saturated_hydraulic_conductivity_factor.value":
                    [0.03, 1.0, 1.0, 1.0, 1.0, 1.0],
                "input.static.soil_water_saturated_zone_top__capillary_rise_max_water_table_depth.value": 4000.0,
                "input.static.soil_water_saturated_zone_top__capillary_rise_averianov_exponent.value": 1,

                # State parameters
                "state.variables.vegetation_canopy_water__depth": "canopystorage",
                "state.variables.soil_water_saturated_zone__depth": "satwaterdepth",
                "state.variables.soil_layer_water_unsaturated_zone__depth": "ustorelayerdepth",
                "state.variables.soil_surface__temperature": "tsoil",
                "state.variables.snowpack_dry_snow__leq_depth": "snow",
                "state.variables.snowpack_liquid_water__depth": "snowwater",
                "state.variables.land_surface_water__instantaneous_volume_flow_rate": "q_land",
                "state.variables.land_surface_water__depth": "h_land",
                "state.variables.river_water__instantaneous_volume_flow_rate": "q_river",
                "state.variables.river_water__depth": "h_river",
                "state.variables.subsurface_water__hydraulic_head": "head",

                # Output parameters
                "output.netcdf_grid.path": "output.nc",
                "output.netcdf_grid.compressionlevel": 0,
                "output.netcdf_grid.variables.river_water__volume_flow_rate": "q_av_river",
                "output.path": "output.nc"
            }

            # Generate configuration section
            config = {
                "setup_config": {
                    "data": data
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
        # Get subbasin river outlet
        subbasin_river_outlet = gpd.read_file(
            self.scenario_and_id_folder.parent / "terrain/river_outlet.shp"
        )

        # Get subbasin river outlet coordinates
        subbasin_river_outlet_coords = list(
            subbasin_river_outlet.geometry.iloc[0].coords
        )[0]

        # Make sure it is in list style (plain floats so the YAML stays clean)
        subbasin_river_outlet_coords_list = [
            float(subbasin_river_outlet_coords[0]), float(subbasin_river_outlet_coords[1])
        ]

        # Generate basemaps section
        basemaps = {
            "setup_basemaps": {
                "hydrography_fn": "merit_hydrox",
                "upscale_method": "ihu",
                "res": self.resolution,
                "region": {
                    "subbasin": subbasin_river_outlet_coords_list,
                    "strord": 4,
                },
                "output_names": {
                    "basin__local_drain_direction": "wflow_ldd",
                    "subbasin_location__count": "wflow_subcatch",
                    "land_surface__slope": "Slope",
                },
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
        # Set up river path
        river_path = self.hydromt_path / "river_data" / self.river_name / f"{self.river_name}.json"

        # Get river information
        with open(river_path, "r", encoding="utf-8") as f:
            river_information = json.load(f)['setup_rivers']

        # Generate rivers section
        rivers = {
            "setup_rivers": {
                "hydrography_fn": "merit_hydrox",
                "river_geom_fn": "hydro_rivers_lin",
                "river_upa": river_information["river_upa"],
                "rivdph_method": "manning",
                "min_rivdph": river_information["min_rivdph"],
                "min_rivwth": river_information["min_rivwth"],
                "slope_len": self.resolution * 3,
                "smooth_len": self.resolution * 5,
                "river_routing": "kinematic_wave",
                "output_names": {
                    "river_location__mask": "wflow_river",
                    "river__length": "wflow_riverlength",
                    "river__width": "wflow_riverwidth",
                    "river_bank_water__depth": "RiverDepth",
                    "river__slope": "RiverSlope",
                },
            }
        }

        return rivers

    def river_roughness_section(self) -> dict:
        """
        Write out setup_river_roughness step

        Returns
        -------
        river_roughness_section : dict
            A dictionary that contains river roughness' section
        """
        # Generate river roughness section
        river_roughness_section = {
            "setup_river_roughness": {
                "output_name": "N_River"
            }
        }

        return river_roughness_section

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
            "setup_reservoirs_no_control": {
                "reservoirs_fn": "hydro_lakes",
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
        match self.landcover:
            case LandcoverClassDataset.GLOBCOVER:
                landcover_mapping = str(self.hydromt_path / "globcover_mapping_modified.csv")
            case LandcoverClassDataset.LCDB:
                landcover_mapping = str(self.hydromt_path / "lcdb_mapping.csv")

        # Generate landuse/landcover's section
        landcover = {
            "setup_lulcmaps": {
                "lulc_fn": "landcover",
                "lulc_mapping_fn": landcover_mapping
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
        # Set up lulc_zero_classes
        if self.landcover == LandcoverClassDataset.GLOBCOVER:
            lulc_zero_classes = [200, 210, 220]
        else:
            lulc_zero_classes = [14, 20, 21, 22]

        # Generate lai section
        lai = {
            "setup_laimaps": {
                "lai_fn": "modis_lai",
                "lulc_fn": "landcover",
                "lulc_sampling_method": "any",
                "lulc_zero_classes": lulc_zero_classes,
                "buffer": 2,
                "output_name": "LAI",
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
                "ptf_ksatver": "brakensiek",
                "wflow_thicknesslayers": [50, 100, 150, 250, 350],
                "output_names": {
                    "soil_water__saturated_volume_fraction": "thetaS",
                    "soil_water__residual_volume_fraction": "thetaR",
                    "soil_surface_water__vertical_saturated_hydraulic_conductivity": "KsatVer",
                    "soil__thickness": "SoilThickness",
                    "soil_water__vertical_saturated_hydraulic_conductivity_scale_parameter": "f",
                    "soil_layer_water__brooks_corey_exponent": "c",
                },
            }
        }

        return soil

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
        # Read Json file to collect some site information
        river_path = self.hydromt_path / f"river_data/{self.river_name}/{self.river_name}.json"
        with open(river_path, "r", encoding="utf-8") as f:
            site = json.load(f)['setup_constant_pars']

        # Generate constant parameters
        constant_parameters = {
            "setup_constant_pars": {
                "subsurface_water__horizontal_to_vertical_saturated_hydraulic_conductivity_ratio": site["KsatHorFrac"],
                "snowpack__degree_day_coefficient": 3.75653,
                "soil_surface_water__infiltration_reduction_parameter": 0.038,
                "vegetation_canopy_water__mean_evaporation_to_mean_precipitation_ratio": 0.11,
                "compacted_soil_surface_water__infiltration_capacity": site["InfiltCapPath"],
                "soil_water_saturated_zone_bottom__max_leakage_volume_flux": 1,
                "soil_wet_root__sigmoid_function_shape_parameter": -500,
                "atmosphere_air__snowfall_temperature_threshold": 0,
                "atmosphere_air__snowfall_temperature_interval": 2,
                "snowpack__melting_temperature_threshold": 0,
                "snowpack__liquid_water_holding_capacity": 0.1,
                "glacier_ice__degree_day_coefficient": 5.3,
                "glacier_firn_accumulation__snowpack_dry_snow_leq_depth_fraction": 0.002,
                "glacier_ice__melting_temperature_threshold": 1.3,
                "vegetation_root__feddes_critical_pressure_head_h1": 0.0,
                "vegetation_root__feddes_critical_pressure_head_h2": -100.0,
                "vegetation_root__feddes_critical_pressure_head_h3_high": -400.0,
                "vegetation_root__feddes_critical_pressure_head_h3_low": -1000.0,
                "vegetation_root__feddes_critical_pressure_head_h4": -16000.0,
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
        if self.polygons is not None:
            # Generate "write" section
            write_section = {
                "staticmaps.write": {},
                "geoms.write": {},
                "config.write": {}
            }
        else:
            # Generate "write" section
            write_section = {
                "forcing.write": {"output_frequency": "D"},  # leave out when polygons is not None
                "staticmaps.write": {},
                "geoms.write": {},
                "config.write": {}
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
        log.info("Setting up wflow build config")
        # Set up wflow build dictionary
        wflow_build = {}

        # Set up sections list
        if self.polygons is not None:
            sections_list = [
                self.config_section(),
                self.landcover_section(),
                self.lai_section(),
                self.write_section()
            ]
        else:
            if str(self.forcing_path).endswith(".nc"):
                sections_list = [
                    self.config_section(),
                    self.basemaps_section(),
                    self.rivers_section(),
                    self.river_roughness_section(),
                    self.lakes_section(),
                    self.landcover_section(),
                    self.lai_section(),
                    self.soil_section(),
                    self.constant_parameters_section(),
                    self.write_section()
                ]
            else:
                sections_list = [
                    self.config_section(),
                    self.basemaps_section(),
                    self.rivers_section(),
                    self.river_roughness_section(),
                    self.lakes_section(),
                    self.landcover_section(),
                    self.lai_section(),
                    self.soil_section(),
                    self.precipitation_section(),
                    self.temperature_section(),
                    self.potential_evaporation_section(),
                    self.constant_parameters_section(),
                    self.write_section()
                ]

        # Generate wflow build steps (hydromt v1: a LIST of single-step dicts)
        steps = []
        for each_section in sections_list:
            if isinstance(each_section, list):  # e.g. write_section() returning a list
                steps.extend(each_section)
            else:
                # split any multi-key dict into one item per step
                for step_name, step_args in each_section.items():
                    steps.append({step_name: step_args})

        wflow_build = {"steps": steps}

        return wflow_build

    def write_out_wflow_build(
        self,
        wflow_build: dict
    ) -> None:
        """
        Write out wflow_build.yml

        Parameters
        ----------
        wflow_build : dict
            A dictionary contains information of all sections
        """
        # Set up output filename
        output_filename = self.scenario_and_id_folder / "hydrological_process/wflow_build.yml"

        log.info(f"Writing out {output_filename}")
        # Generate content for wflow_build.yml
        with open(output_filename, "w", encoding="utf-8") as output_file:
            yaml.dump(
                wflow_build,
                output_file,
                Dumper=_IndentedDumper,
                sort_keys=False,
                width=200,
            )

    def wflow_build_generator(self) -> None:
        """Generate data_catalog.yml file"""
        # Set up content for wflow_build file
        wflow_build = self.wflow_build_section()

        # Write wflow_build file
        self.write_out_wflow_build(wflow_build)
