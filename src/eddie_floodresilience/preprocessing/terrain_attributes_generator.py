"""
Generate terrain attributes by processing DEM and roughness data
mainly using Whitebox package.
"""

from pathlib import Path
from whitebox_workflows import WbEnvironment, Raster
from whitebox.whitebox_tools import WhiteboxTools
from osgeo import gdal # Import gdal before rasterio
import xarray as xr
import rioxarray as rxr
import numpy as np
from scipy.ndimage import distance_transform_edt
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
from shapely.geometry import LineString, MultiLineString


# Create whitebox environment and whitebox tools
wbe = WbEnvironment()
wbe.verbose = True
wbe.max_procs = -1
wbt = WhiteboxTools()


class TerrainAttributesGenerator():
    """
    This class is to generate terrain attributes
    for generating stream attributes
    """

    def __init__(
            self,
            path: Path,
            raster_name: str = 'dem',
            resolution: float = 50,
            threshold: int = 1000,
            origin_filename: str = '8m_geofabric'
    ) -> None:
        """
        Declare variables to be used in later functions

        Parameters
        ----------
        path: Path
            Path to the directory that contains necessary files to generate terrain data
        raster_name: str = 'dem'
            Name of the raster. Mostly 'dem' and 'roughness'
        resolution: float = 100
            Resolution to resample data. Default is 100m in crs 2193
        threshold: int = 1000
            Minimum number of cells/up-slope area required to initiate and main a channel.
            Default is 1000
        origin_filename : str = '8m_geofabric'
            Name of terrain raster filename.
            At the moment, only two names - 8m_geofabric and 4m_geofabric
        """
        self.path = path
        self.raster_name = raster_name
        self.resolution = resolution
        self.threshold = threshold
        self.origin_filename = origin_filename

    def raster_resampling(
            self,
            resampling_method: str = 'nn'
    ) -> None:
        """
        Resample raster to a specific resolution (good with GeoTiff file)

        Parameters
        -----------
        resampling_method: str = 'nn'
            Resampling methods includes "nn" (nearest neighbor), 'bi-linear',
            and 'cc' (cubic convolution). Default is 'nn'
        """
        output_path = self.path / f"{self.origin_filename}_{self.raster_name}_for_wflow_coarser.tif"
        input_path = self.path / f"{self.origin_filename}_{self.raster_name}_for_wflow.tif"

        if not output_path.is_file():
            # Resample raster
            wbt.resample(
                inputs=str(input_path),
                output=str(output_path),
                cell_size=self.resolution,
                method=resampling_method
            )
        else:
            print(f"'{output_path.name}' already exists!")

    def raster_fill_depression(
            self,
            flat_increment: float = 0.0001
    ) -> None:
        """
        Fill depressions in raster (specifically in DEM)

        Parameters
        ----------
        flat_increment: float = 1
            If flat surfaces such as lakes have the slope 0 it will act like a sink.
            This parameter will set a small slope to flat areas. Default is 1.
            https://github.com/williamlidberg/Whitebox-tutorial/blob/main/streams.py
        """
        input_path = self.path / f"{self.origin_filename}_{self.raster_name}_for_wflow_coarser.tif"
        output_path_no_deps = self.path / f"{self.origin_filename}_{self.raster_name}_for_wflow_coarser_nodeps.tif"
        output_path_crs = self.path / f"{self.origin_filename}_{self.raster_name}_for_wflow_coarser_nodeps_crs.tif"

        if not output_path_crs.is_file():
            # Read the raster using whitebox tool
            raster_no_deps = wbe.read_raster(str(input_path))

            # Fill depressions in the raster
            raster_no_deps = wbe.fill_depressions(
                raster_no_deps,
                flat_increment=flat_increment
            )

            # Write out
            wbe.write_raster(
                raster_no_deps,
                str(output_path_no_deps),
                compress=False
            )

            # Read data using rioxarray to add crs and then overwrite out
            with rxr.open_rasterio(output_path_no_deps) as raster_no_deps_crs:
                raster_no_deps_crs = raster_no_deps_crs.rio.write_crs('EPSG:2193')
                raster_no_deps_crs.rio.to_raster(fr"{self.path}\{self.origin_filename}_{self.raster_name}_for_wflow_coarser_nodeps_crs.tif")
        else:
            print(f"'{output_path_crs.name}' already exists!")

    def d8_pointer_generator(self) -> None:
        """
        Generate D8 pointers based on D8 algorithm (O'Callaghan and Mark, 1984) (mainly from DEM)
        https://www.whiteboxgeo.com/manual/wbw-user-manual/book/tool_help.html#d8_pointer
        """
        input_path = self.path / f"{self.origin_filename}_{self.raster_name}_for_wflow_coarser_nodeps_crs.tif"
        output_path = self.path / f"{self.origin_filename}_d8_pointer.tif"

        if not output_path.is_file():
            # Read the raster using whitebox tool
            raster_no_deps = wbe.read_raster(str(input_path))

            # Generate D8 pointer
            d8_pointer = wbe.d8_pointer(raster_no_deps)

            # Write out raster
            wbe.write_raster(
                d8_pointer,
                str(output_path)
            )
        else:
            print(f"'{output_path.name}' already exists!")

    def d8_stream_generator(
            self,
            catchment_area: bool = True
    ) -> None:
        """
        Generate D8 flow accumulation based on D8 algorithm (O'Callaghan and Mark, 1984) (mainly from DEM)

        Parameters
        -----------
        catchment_area: bool = True
            If True, flow accumulation under catchment are format will be added
            If False, only flow accumulation under cell format will be generated
        """
        input_path = self.path / f"{self.origin_filename}_{self.raster_name}_for_wflow_coarser_nodeps_crs.tif"
        flow_path_acc_cells = self.path / f"{self.origin_filename}_flow_acc_d8_cells.tif"
        streams_path = self.path / f"{self.origin_filename}_streams_d8.tif"
        flow_path_acc_area = self.path / f"{self.origin_filename}_flow_acc_d8_area_m2.tif"

        if not streams_path.is_file():
            # Generate D8 flow accumulation - output is 'cell' type
            wbt.d8_flow_accumulation(
                i=str(input_path),
                out_type='cells',
                output=str(flow_path_acc_cells)
            )

            # Extract streams from the flow accumulation
            wbt.extract_streams(
                flow_accum=str(flow_path_acc_cells),
                output=str(streams_path),
                threshold=self.threshold
            )
        else:
            print(f"'{streams_path.name}' already exists!")

        if catchment_area:
            if not flow_path_acc_area.is_file():
                # Generate D8 flow accumulation - output is 'catchment area' type
                wbt.d8_flow_accumulation(
                    i=str(input_path),
                    out_type="catchment area",
                    output=str(flow_path_acc_area)
                )
            else:
                print(f"'{flow_path_acc_area.name}' already exists!")

    def strahler_stream_order_generator(self) -> None:
        """
        Generate Strahler stream order based on Strahler algorithm (Strahler, A. N., 1957)
        and convert to vector
        https://www.whiteboxgeo.com/manual/wbt_book/available_tools/stream_network_analysis.html#strahlerstreamorder
        https://www.whiteboxgeo.com/manual/wbt_book/available_tools/stream_network_analysis.html?highlight=extract%20stream#RasterStreamsToVector
        """
        # Generate stream order
        wbt.strahler_stream_order(
            d8_pntr=str(self.path / f"{self.origin_filename}_d8_pointer.tif"),
            streams=str(self.path / f"{self.origin_filename}_streams_d8.tif"),
            output=str(self.path / f"{self.origin_filename}_strahler_d8.tif")
        )

        # Convert stream raster to vector shapefile
        wbt.raster_streams_to_vector(
            d8_pntr=str(self.path / f"{self.origin_filename}_d8_pointer.tif"),
            streams=str(self.path / f"{self.origin_filename}_streams_d8.tif"),
            output=str(self.path / f"{self.origin_filename}_streams_d8.shp")
        )

    def streams_repairer(
            self, 
            info_from_watershed: bool = False
        ) -> None:
        """
        Repair streams if they are not connected
        
        Parameters
        ----------
        info_from_watershed : bool = False
            The streams contain watershed information. Default is False.
            If False, it is just normal D8 streams.
            If True, it is D8 streams that includes watershed information
        """
        if info_from_watershed == False:
            # Repair streams within 50m distance
            wbt.repair_stream_vector_topology(
                i=str(self.path / f"{self.origin_filename}_streams_d8.shp"),
                output=str(self.path / f"{self.origin_filename}_repaired_streams_d8.shp"),
                dist="50"
            )
            
            # Read the repaired streams
            repaired_streams = gpd.read_file(
                self.path / f"{self.origin_filename}_repaired_streams_d8.shp",
                on_invalid='ignore'  # Ignore broken geometries            
            )
            
            # Output filename
            output_filename = f"{self.origin_filename}_filtered_repaired_streams_d8.shp"
            
        else:
            # Repair streams within 0.0005 degree or ~50m distance
            wbt.repair_stream_vector_topology(
                i=str(self.path / f"{self.origin_filename}_streams_watershed_more_info.shp"),
                output=str(self.path / f"{self.origin_filename}_repaired_streams_watershed_more_info.shp"),
                dist="50"
            )
            
            # Read the repaired streams
            repaired_streams = gpd.read_file(
                self.path / f"{self.origin_filename}_repaired_streams_watershed_more_info.shp",
                on_invalid='ignore'  # Ignore broken geometries            
            )
            
            # Output filename
            output_filename = f"{self.origin_filename}_filtered_repaired_streams_watershed_more_info.shp"
        
        # Remove all Polygons
        # This should be improved in the future
        repaired_streams = repaired_streams[
            repaired_streams.geometry.apply(
                lambda g: isinstance(g, (LineString, MultiLineString))    
            )    
        ]
        
        # Remove very short lines
        min_line = 1
        repaired_streams = repaired_streams[
            repaired_streams.geometry.length > min_line
        ]
        
        # Set up crs
        # This will be changed in the future
        repaired_streams = repaired_streams.set_crs(crs=2193)
        
        # Write out the repaired streams again
        repaired_streams.to_file(self.path / output_filename)

    def raster_to_points_dataframe(
            self,
            file_name: str,
            column_name: str
    ) -> gpd.GeoDataFrame:
        """
        Convert raster values of pixels in stream network (could be area or strahler) to points

        Parameters
        -----------
        file_name : str
            Name of the file that will be used to convert to points
        column_name : str
            Name of column that is converted from 'VALUE1' to
            Options are mostly 'upstream_area_m2' and 'strahler'

        Returns
        ---------
        points_df : gpd.GeoDataFrame
            A GeoDataFrame contains points data of 'upstream_area_m2' or 'strahler'
        """
        input_path = self.path / f"{self.origin_filename}_streams_d8.tif"
        output_path = self.path / f"{self.origin_filename}_stream_pixels_pts.shp"
        file_path = self.path / f"{file_name}.tif"

        if file_name != 'strahler_d8':
            # Convert raster to point shapefile
            wbt.raster_to_vector_points(
                i=str(input_path),
                output=str(output_path)
            )

        # Convert raster to vector of point shape type from flow accumulation
        wbt.extract_raster_values_at_points(
            str(file_path),
            points=str(output_path)
        )

        # Convert to geopandas dataframe
        points_df = gpd.read_file(output_path)
        points_df = points_df.rename(columns={'VALUE1': f'{column_name}'})

        return points_df

    def roughness_to_manning(
            self,
            roughness: xr.DataArray,
            h: float = 1
    ) -> None:
        """
        Convert raster of roughness to manning's n

        Parameters
        ----------
        roughness : Any
            A raster of roughness data
        h : float = 1
            Value of depth. Default is 1
        """
        # Convert roughness length to Manning's n
        ratio_h_roughness = h / roughness
        numerator = 0.41 * (h ** (1 / 6)) * (ratio_h_roughness - 1)
        denominator = np.sqrt(9.80665) * (1 + ratio_h_roughness * (np.log(ratio_h_roughness) - 1))
        manning_n = numerator / denominator

        output_path = self.path / f"{self.origin_filename}_streams_manning.tif"
        # Write out Manning's n
        manning_n.rio.to_raster(str(output_path))


class StreamTopologyGenerator():
    """This class is to generate stream topology data"""

    def __init__(
            self,
            path: Path,
            resolution: float = 50,
            threshold: int = 1000,
            origin_filename: str = '8m_geofabric'
    ) -> None:
        """
        Generate stream topology: 'upstream_area_m2' and 'strahler'

        Parameters
        ----------
        path: str
            Path to the directory that contains necessary files to generate stream data
        resolution: float = 100
            Resolution to resample data. Default is 100m
        threshold: int = 1000
            Minimum number of cells/up-slope area required to initiate and main a channel.
            Default is 1000
        origin_filename : str = '8m_geofabric'
            Name of terrain raster filename.
            At the moment, only two names - 8m_geofabric and 4m_geofabric
        """
        self.path = path
        self.resolution = resolution
        self.threshold = threshold
        self.origin_filename = origin_filename
        
        self.stream_topology_data = TerrainAttributesGenerator(self.path, 'dem')

    def merge_stream_topology_points(self) -> gpd.GeoDataFrame:
        """
        Generate point dataframe that merges upstream catchment area and strahler order.
        This dataframe already removes the rows with same FIDs.

        Returns
        --------
        agg_upstream_area_and_strahler : gpd.GeoDataFrame
            A geopandas dataframe that contains both 'upstream_area_m2' and 'strahler'.
            There is no rows with the same FIDs
        """
        # Generate points that include upstream area under geopandas dataframe
        points_area_m2 = self.stream_topology_data.raster_to_points_dataframe(
            f'{self.origin_filename}_flow_acc_d8_area_m2',
            'upstream_area_m2'
        )

        # Generate points that include strahler order under geopandas dataframe
        points_strahler_order = self.stream_topology_data.raster_to_points_dataframe(
            f'{self.origin_filename}_strahler_d8',
            'strahler'
        )

        # Merge two points geopandas dataframe
        points_merge = points_area_m2.merge(
            points_strahler_order[['FID', 'strahler']],
            on='FID',
            how='left'
        )

        # After raster-to-point conversions and merging, duplicate FIDs are produced
        # which leads to multiple rows with the same FID in the merged dataframe.
        # Hence, the agg function is used to select the max values for both
        # 'upstream_area_m2' and 'strahler'.
        # - For 'upstream_area_m2', as flow accumulation
        # increases downstream, the largest upstream area corresponds to the true
        # accumulate catchment area at that point. Hence, the maximum upstream area
        # preserves the most hydrologically meaningful value.
        # - For 'strahler', as strahler order increases when tributaries merge,
        # the maximum order ensures the feature keeps the correct stream hierarchy classification.
        agg_upstream_area_and_strahler = (
            points_merge.groupby('FID').agg(
                upstream_area=('upstream_area_m2', 'max'),
                strahler=('strahler', 'max')
            )
            .reset_index()
        )

        return agg_upstream_area_and_strahler

    def merge_upstream_area_strahler_stream_geometry(
            self,
            agg_upstream_area_and_strahler: gpd.GeoDataFrame
    ) -> None:
        """
        Merge dataframes of 'upstream_area_m2' and 'strahler' with 'geometry' of stream using FID
        and write out the merged dataframe

        Parameters
        -----------
        agg_upstream_area_and_strahler: gpd.GeoDataFrame
            Geopandas dataframe that contains stream attributes "upstream_area_m2" and "strahler"
        """
        # Read filtered and repaired D8 stream dataframe
        stream_input_path = self.path / f"{self.origin_filename}_filtered_repaired_streams_d8.shp"
        streams = gpd.read_file(stream_input_path)

        # Merge the aggregated dataframe with stream dataframe
        streams = streams.merge(
            agg_upstream_area_and_strahler,
            on='FID',
            how='left'
        )

        # Convert from km2 to m2
        streams['upstream_area'] = streams['upstream_area'] * 1e6

        # Rename columns
        streams_rename = streams.rename(
            columns={
                'upstream_area': 'uparea',
                'strahler': 'strord'
            }
        )

        # Add crs
        streams_rename = streams_rename.set_crs(2193)

        # Write out
        stream_output_path = self.path / f"{self.origin_filename}_streams_d8_area_strahler.shp"
        streams_rename.to_file(stream_output_path)

    def dataframe_upstream_area_strahler_geometry_generator(
            self,
    ) -> None:
        """Generate geodataframe of 'upstream_area_m2' and 'strahler'"""
        # Resample raster
        self.stream_topology_data.raster_resampling('nn')

        # Fill depression
        self.stream_topology_data.raster_fill_depression(0.0001)

        # Generate D8 pointer
        self.stream_topology_data.d8_pointer_generator()

        # Generate D8 stream generator
        self.stream_topology_data.d8_stream_generator(catchment_area=True)

        # Generate strahler order
        self.stream_topology_data.strahler_stream_order_generator()
        
        # Repair streams
        self.stream_topology_data.streams_repairer(info_from_watershed=False)

        # Collect dataframe of 'upstream_area_m2' and 'strahler'
        df_upstream_area_strahler = self.merge_stream_topology_points()

        # Merge dataframe of 'upstream_area_m2' and 'strahler'
        # with dataframe of stream geometry and write out
        self.merge_upstream_area_strahler_stream_geometry(df_upstream_area_strahler)


class StreamHydraulicsGenerator():
    """This class is to generate hydraulic stream attributes"""

    def __init__(
            self,
            path: Path,
            outlet_gauge_locations_file: str,
            streams_bankfull_stage: float = 1.5,
            resolution: float = 50,
            threshold: int = 1000,
            width_rate_control: float = 2,
            discharge_rate_control: float = 1,
            origin_filename: str = '8m_geofabric'
    ) -> None:
        """
        Generate hydraulic stream attributes

        Parameters
        -----------
        path : Path
            Common path to directory that stores necessary file for generating hydraulic stream data
        outlet_gauge_locations_file : str
            Filename that contains locations of outlet and gauges
        streams_bankfull_stage : float = 1.5
            The stage to focus on the area that is considered as stream/river area
            or bankfull area comparing with HAND.
            Default is 1.5
        resolution : float = 100
            Resolution to resample data
        threshold : int = 1000
            Minimum number of cells/up-slope area required to initiate and main a channel.
            Default is 1000
        width_rate_control : float = 2
            The rate to control river width. Default is 2
        discharge_rate_control: float = 1
            The rate to control river discharge. Default is 1
        origin_filename : str = '8m_geofabric'
            Name of terrain raster filename.
            At the moment, only two names - 8m_geofabric and 4m_geofabric
        """
        self.path = path
        self.streams_bankfull_stage = streams_bankfull_stage
        self.resolution = resolution
        self.threshold = threshold
        self.width_rate_control = width_rate_control
        self.discharge_rate_control = discharge_rate_control        
        self.outlet_gauge_locations_file = outlet_gauge_locations_file
        self.stream_topology_data = TerrainAttributesGenerator(
            self.path, 
            'dem',
            self.resolution,
            self.threshold
        )
        self.roughness_data = TerrainAttributesGenerator(
            self.path, 
            'roughness',
            self.resolution,
            self.threshold
        )
        self.origin_filename = origin_filename

    def watershed_generator(
            self,
            snap_dist: float = 5.0,
            filter_size: int = 5
    ) -> Raster:
        """
        Generate watershed based on D8 pointer (flow direction) and list of points of outlet and gauges

        Parameters
        ----------
        snap_dist : float = 5.0
            Measures in map units (e.g. meters, default is meters) the given maximum distance
            between the pour points to the location coincident with the nearest stream cell.
            Default is 5 meters.
            https://www.whiteboxgeo.com/manual/wbt_book/available_tools/hydrological_analysis.html#JensonSnapPourPoints
        filter_size : int = 5
            Filter size to smooth a vector coverage of either a Polyline or Polygon base.
            It can be any integer larger than or equal to 3. Default here is 5.
            https://www.whiteboxgeo.com/manual/wbw-user-manual/book/tool_help.html#smooth_vectors

        Returns
        -------
        outlet_watershed : Raster
            A raster showing watershed within DEM
        """
        # Read stream raster
        stream_path = self.path / f"{self.origin_filename}_streams_d8.tif"
        streams = wbe.read_raster(str(stream_path))

        # Read D8 pointer
        d8_pointer_path = self.path / f"{self.origin_filename}_d8_pointer.tif"
        d8_pointer = wbe.read_raster(str(d8_pointer_path))

        # Extract watershed for specific points of outlet and gauges
        outlet_gauge_points_path = self.path / f"{self.outlet_gauge_locations_file}.shp"
        outlet_gauge_points = wbe.read_vector(str(outlet_gauge_points_path))

        # Ensure the watershed or streamlines that have points of outlet and gauges
        outlet_gauge_points_on_streams = wbe.jenson_snap_pour_points(
            outlet_gauge_points,
            streams,
            snap_dist=snap_dist
        )

        # Extract watershed of the outlet
        outlet_watershed = wbe.watershed(
            d8_pointer=d8_pointer,
            pour_points=outlet_gauge_points_on_streams
        )

        # Write out watershed raster
        watershed_path = self.path / f"{self.origin_filename}_watershed.tif"
        wbe.write_raster(
            outlet_watershed,
            str(watershed_path),
            compress=False
        )

        # Generate watershed polygon for checking (if necessary)
        watershed_polygon = wbe.raster_to_vector_polygons(outlet_watershed)

        # Smooth the watershed map
        watershed_polygon = wbe.smooth_vectors(
            watershed_polygon,
            filter_size=filter_size
        )

        # Write out
        watershed_polygon_path = self.path / f"{self.origin_filename}_watershed.shp"
        wbe.write_vector(
            watershed_polygon,
            str(watershed_polygon_path)
        )

        return outlet_watershed

    def stream_watershed_raster_generator(
            self,
            outlet_watershed: Raster
    ) -> tuple[Raster, Raster]:
        """
        Generate streams within watershed that contributes to the outlet

        Parameters
        ----------
        outlet_watershed: Raster
            A raster shows watershed within DEM

        Returns
        -------
        streams_watershed : Raster
            A raster of watershed that contributes to the outlet
        streams_watershed_raster : Raster
            A raster of  watershed that contributes to the outlet
            and contains more information
        """
        # Read stream raster
        stream_path = self.path / f"{self.origin_filename}_streams_d8.tif"
        streams = wbe.read_raster(str(stream_path))

        # Read D8 pointer raster
        d8_pointer_path = self.path / f"{self.origin_filename}_d8_pointer.tif"
        d8_pointer = wbe.read_raster(str(d8_pointer_path))

        # Read DEM that its depressions are filled
        dem_no_deps_path = self.path / f"{self.origin_filename}_dem_for_wflow_coarser_nodeps_crs.tif"
        dem_no_deps = wbe.read_raster(str(dem_no_deps_path))

        # Filter to select only streams inside the watershed
        streams_watershed = streams * outlet_watershed

        # Write out raster with streams within watershed
        # (this stream data just has 1 and 0 values)
        stream_path_watershed = self.path / f"{self.origin_filename}_streams_watershed.tif"
        wbe.write_raster(
            streams_watershed,
            str(stream_path_watershed),
            compress=False
        )

        # Convert stream raster within watershed to vector (just geometry)
        streams_watershed_vector = wbe.raster_streams_to_vector(
            streams_watershed,
            d8_pointer
        )

        # Add more information into stream vector such as
        # reach IDs or FIDs, connectivity, stream orders, flow connectivity information, etc.
        streams_watershed_vector_more_info, _, _, _ = wbe.vector_stream_network_analysis(
            streams_watershed_vector,
            dem_no_deps
        )

        # Write out vector with streams with watershed
        # (this stream data has more information like FIDs, connectivity, etc.)
        stream_path_watershed_more_info = self.path / f"{self.origin_filename}_streams_watershed_more_info.shp"
        wbe.write_vector(
            streams_watershed_vector_more_info,
            str(stream_path_watershed_more_info)
        )
        
        # Repair streams
        self.stream_topology_data.streams_repairer(info_from_watershed=True)
        
        # Read the repaired streams
        repaired_streams_watershed_more_info_path = self.path / f"{self.origin_filename}_filtered_repaired_streams_watershed_more_info.shp"
        repaired_streams_watershed_more_info = wbe.read_vector(
            str(repaired_streams_watershed_more_info_path)
        )

        # Convert back to raster once collecting FID
        streams_watershed_raster = wbe.vector_lines_to_raster(
            repaired_streams_watershed_more_info,
            'FID',
            base_raster=dem_no_deps,
            zero_background=True
        )

        return streams_watershed, streams_watershed_raster

    def hand_generator(self) -> None:
        """Generate height above nearest drainage (HAND)"""
        # Read streams within watershed
        stream_watershed = self.path / f"{self.origin_filename}_streams_watershed.tif"
        streams_watershed = wbe.read_raster(str(stream_watershed))

        # Read DEM that its depressions are filled
        dem_no_deps_path = self.path / f"{self.origin_filename}_dem_for_wflow_coarser_nodeps_crs.tif"
        dem_no_deps = wbe.read_raster(str(dem_no_deps_path))

        # Calculate HAND
        hand = wbe.elevation_above_stream(
            dem_no_deps,
            streams_watershed
        )

        # Write out
        hand_path = self.path / f"{self.origin_filename}_hand.tif"
        wbe.write_raster(
            hand,
            str(hand_path),
            compress=False
        )

    def stream_bankfull_width_raster_generator(self) -> None:
        """Generate stream bankfull width raster"""
        # Read streams within watershed using rioxarray
        stream_path_watershed = self.path / f"{self.origin_filename}_streams_watershed.tif"
        streams_watershed = rxr.open_rasterio(stream_path_watershed).squeeze()

        # Read HAND raster
        hand_path = self.path / f"{self.origin_filename}_hand.tif"
        hand = rxr.open_rasterio(hand_path).squeeze()

        # Set up bankfull
        bankfull = hand <= self.streams_bankfull_stage

        # Get values of bankfull and stream
        # bankfull_np tells us where the river is
        # stream_np tells us where the streamline is
        bankfull_np = bankfull.values  # Raster where river area = 1 (True) and land = 0 (False)
        # Generate a mask of stream centreline pixels (True = this pixel is part of the stream)
        stream_np = streams_watershed.values > 0

        # Calculate distance
        # Get pixel size: Each pixel represents X meters on the ground
        pixel_size = abs(hand.rio.resolution()[0])

        # Measure distance to the river bank:
        # For every river pixel, it calculates the distance to the nearest non-river pixel (the river bank)
        # It here is the function "distance_transform_edt"
        # So pixels near the bank --> small distance
        # Pixels near the center of the river --> larger distance
        # For example: bank 1m 2m 3m 4m 3m 2m 1m bank
        # (so near the bank --> small distance, near the middle --> larger distance)
        # ==> Logic is: How far am I from the river edge?
        distance = distance_transform_edt(bankfull_np) * pixel_size

        # Filter out only the distance from the center line to a bank
        # So it will be like: bank Nan Nan 4m Nan Nan bank
        # And then double it for the other bank
        bankfull_width = distance[stream_np] * self.width_rate_control

        # Put widths back into a raster
        streams_bankfull_width = np.full(stream_np.shape, np.nan)
        streams_bankfull_width[stream_np] = bankfull_width

        # Convert numpy array to xarray data array
        streams_bankfull_width_da = xr.DataArray(
            streams_bankfull_width,
            coords=hand.coords,
            dims=hand.dims,
            name="bankfull_width"
        )

        # Add crs
        streams_bankfull_width_da.rio.write_crs(
            hand.rio.crs,
            inplace=True
        )

        # Write out
        stream_path_bankfull_width = self.path / f"{self.origin_filename}_streams_bankfull_width.tif"
        streams_bankfull_width_da.rio.to_raster(str(stream_path_bankfull_width))

    def read_streams_watershed_more_info(self) -> gpd.GeoDataFrame:
        """
        Read the file that contains streams' information within watershed

        Returns
        -------
        streams_watershed_vector_more_info : gpd.GeoDataFrame
            Converted-to-vector streams within watershed with more information
        """
        stream_path_watershed_vector_more_info = self.path / f"{self.origin_filename}_filtered_repaired_streams_watershed_more_info.shp"
        streams_watershed_vector_more_info = gpd.read_file(stream_path_watershed_vector_more_info)

        return streams_watershed_vector_more_info

    def buffer_streams_watershed(
            self,
            streams_watershed_vector_more_info: gpd.GeoDataFrame
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Buffer the stream linestrings to capture stream pixels

        Returns
        -------
        streams_watershed_vector_more_info : gpd.GeoDataFrame
            Converted-to-vector streams within watershed with more information
        streams_watershed_buffer : gpd.GeoDataFrame
            Buffered streams vector within watershed that can intersect with pixels
        """
        # Read HAND raster
        hand = rxr.open_rasterio(fr"{self.path}\{self.origin_filename}_hand.tif").squeeze()

        # Buffer by half raster pixel
        # To explain, linestrings have no width and might not be aligned well with the raster.
        # Hence, buffering helps to ensure the stream intersects well with the raster
        pixel_size = abs(hand.rio.resolution()[0])
        streams_watershed_buffer = streams_watershed_vector_more_info.copy()
        streams_watershed_buffer['geometry'] = streams_watershed_buffer.geometry.buffer(pixel_size / 2)

        return streams_watershed_buffer

    def assign_streams_hydraulic_values(
            self,
            streams_watershed_vector_more_info: gpd.GeoDataFrame,
            streams_watershed_buffer: gpd.GeoDataFrame,
            hydraulic_name: str
    ) -> None:
        """
        Assign stream hydraulic values to each stream segment in the vector network

        Parameters
        -----------
        streams_watershed_vector_more_info: gpd.GeoDataFrame
            Converted-to-vector streams within watershed with more information
        streams_watershed_buffer: gpd.GeoDataFrame
            Buffered streams vector within watershed that can intersect with pixels
        hydraulic_name: str
            Name of hydraulic stream attributes
        """
        # Extract stream hydraulic values for each reach
        stream_hydraulic_path_values = self.path / f"{self.origin_filename}_streams_{hydraulic_name}.tif"
        streams_hydraulic_values = zonal_stats(
            vectors=streams_watershed_buffer,
            raster=str(stream_hydraulic_path_values),
            stats=['mean'],
            all_touched=True  # includes all pixels by touching the buffered ones
        )

        # Convert list of dicts to dataframe
        streams_hydraulic_values_df = pd.DataFrame(streams_hydraulic_values)

        # Add stream hydraulic values to streams watershed dataframe
        streams_hydraulic_values_linestring = streams_watershed_vector_more_info.join(
            streams_hydraulic_values_df
        )
        streams_hydraulic_values_linestring[f'{hydraulic_name}'] = streams_hydraulic_values_linestring['mean']

        # Write out
        stream_hydraulic_path_linestring = self.path / f"{self.origin_filename}_streams_{hydraulic_name}_linestring.shp"
        streams_hydraulic_values_linestring.to_file(str(stream_hydraulic_path_linestring))

    def stream_bankfull_width_linestring_generator(self) -> None:
        """Generate streams' bankfull width vector"""
        # Read file that contains streams' information within watershed
        streams_watershed_vector_more_info = self.read_streams_watershed_more_info()

        # Buffer stream linestring to capture stream pixels
        streams_watershed_buffer = self.buffer_streams_watershed(
            streams_watershed_vector_more_info
        )

        # Assign stream bankfull width to stream linestring
        self.assign_streams_hydraulic_values(
            streams_watershed_vector_more_info,
            streams_watershed_buffer,
            'bankfull_width'
        )

    def stream_manning_linestring_generator(self) -> None:
        """Generate streams' manning's n"""
        # Resample roughness raster
        self.roughness_data.raster_resampling('nn')

        # Read coarse roughness raster
        roughness_path = self.path / f"{self.origin_filename}_roughness_for_wflow_coarser.tif"
        roughness_for_wflow_coarser = rxr.open_rasterio(str(roughness_path))

        # Convert roughness to manning
        self.roughness_data.roughness_to_manning(
            roughness_for_wflow_coarser,
            1
        )

        # Read file that contains streams' information within watershed
        streams_watershed_vector_more_info = self.read_streams_watershed_more_info()

        # Buffer stream linestring to capture stream pixels
        streams_watershed_buffer = self.buffer_streams_watershed(
            streams_watershed_vector_more_info
        )

        # Assign stream bankfull width to stream linestring
        self.assign_streams_hydraulic_values(
            streams_watershed_vector_more_info,
            streams_watershed_buffer,
            'manning'
        )

    def stream_slope_linestring_generator(self) -> None:
        """Generate streams' slopes"""
        # Read DEM that its depressions are filled
        dem_no_deps_path = self.path / f"{self.origin_filename}_dem_for_wflow_coarser_nodeps_crs.tif"
        dem_no_deps = wbe.read_raster(str(dem_no_deps_path))

        # Generate slope from DEM
        slope = wbe.slope(
            dem_no_deps,
            units='percent'
        )

        # Write out slope
        slope_path = self.path / f"{self.origin_filename}_streams_slope.tif"
        wbe.write_raster(
            slope,
            str(slope_path),
            compress=False
        )

        # Read file that contains streams' information within watershed
        streams_watershed_vector_more_info = self.read_streams_watershed_more_info()

        # Buffer stream linestring to capture stream pixels
        streams_watershed_buffer = self.buffer_streams_watershed(
            streams_watershed_vector_more_info
        )

        # Assign stream bankfull width to stream linestring
        self.assign_streams_hydraulic_values(
            streams_watershed_vector_more_info,
            streams_watershed_buffer,
            'slope'
        )

    def bankfull_discharge_calculation(
            self,
            streams_bankfull_width: pd.Series,
            streams_slope: pd.Series,
            streams_manning: pd.Series
    ) -> pd.Series:
        """
        Generate calculation method of streams' bankfull discharge

        Parameters
        ----------
        streams_bankfull_width: gdp.GeoDataFrame
            Streams' bankfull width vector
        streams_slope: gdp.GeoDataFrame
            Streams' slope vector
        streams_manning: gdp.GeoDataFrame
            Streams' manning's n vector

        Returns
        -------
        bankfull_discharge : gdp.GeoDataFrame
            Stream's bankfull discharge values
        """
        # Calculate cross-sectional area (rectangular)
        cross_sectional_area = streams_bankfull_width * self.streams_bankfull_stage

        # Calculate wetted perimeter
        wetted_perimeter = streams_bankfull_width + 2 * self.streams_bankfull_stage

        # Calculate hydraulic radius
        hydraulic_radius = cross_sectional_area / wetted_perimeter

        # Calculate bankfull discharge
        streams_manning_calc = 1 / streams_manning
        hydraulic_radius_calc = hydraulic_radius ** (2 / 3)
        bankfull_discharge = streams_manning_calc * wetted_perimeter * hydraulic_radius_calc * np.sqrt(streams_slope)

        return bankfull_discharge

    def stream_bankfull_discharge_generator(self) -> pd.Series:
        """
        Generate streams' bankfull discharge

        Returns
        -------
        streams_bankfull_discharge : pd.Series
            Streams' bankfull discharge
        """
        # Directories
        stream_path_bankfull_width = self.path / f"{self.origin_filename}_streams_bankfull_width_linestring.shp"
        stream_path_manning = self.path / f"{self.origin_filename}_streams_manning_linestring.shp"
        stream_path_slope = self.path / f"{self.origin_filename}_streams_slope_linestring.shp"

        # Read stream bankfull width shapefile
        streams_bankfull_width = gpd.read_file(stream_path_bankfull_width).bankfull_w
        streams_manning = gpd.read_file(stream_path_manning).manning
        streams_slope = gpd.read_file(stream_path_slope).slope

        # Generate stream bankfull discharge
        streams_bankfull_discharge = self.bankfull_discharge_calculation(
            streams_bankfull_width,
            streams_slope,
            streams_manning
        ) * self.discharge_rate_control

        return streams_bankfull_discharge

    def stream_bankfull_width_discharge_generator(self) -> None:
        """Generate streams' bankfull width and discharge"""
        # Generate stream bankfull discharge
        streams_bankfull_discharge = self.stream_bankfull_discharge_generator()

        # Read stream bankfull width
        stream_path_bankfull_width = self.path / f"{self.origin_filename}_streams_bankfull_width_linestring.shp"
        streams_bankfull_width = gpd.read_file(str(stream_path_bankfull_width))

        # Rename stream bankfull width
        streams_bankfull_width_discharge = streams_bankfull_width.rename(columns={'bankfull_w': 'rivwth'})

        # Add stream bankfull discharge
        streams_bankfull_width_discharge['qbankfull'] = streams_bankfull_discharge

        # Write out with "rivers_lin2019_v1.gpkg"
        # Here we keep the name like that for basic automation
        stream_path_bankfull_width_discharge = self.path / "rivers_lin2019_v1.gpkg"
        streams_bankfull_width_discharge.to_file(stream_path_bankfull_width_discharge)

    def dataframe_stream_bankfull_width_discharge_generator(self) -> None:
        """Generate geopandas dataframe that contains streams' bankfull width and discharge"""
        # Resample raster
        self.stream_topology_data.raster_resampling('nn')

        # Fill depression
        self.stream_topology_data.raster_fill_depression(0.0001)

        # Generate D8 pointer
        self.stream_topology_data.d8_pointer_generator()

        # Generate D8 stream generator
        self.stream_topology_data.d8_stream_generator(catchment_area=True)
                                                      
        # Generate watershed raster
        watershed_polygon = self.watershed_generator(
            5, 5
        )
        self.stream_watershed_raster_generator(
            watershed_polygon
        )

        # Generate HAND raster
        self.hand_generator()

        # Generate stream bankfull width raster
        self.stream_bankfull_width_raster_generator()

        # Generate stream bankfull width linestring
        self.stream_bankfull_width_linestring_generator()

        # Generate stream manning
        self.stream_manning_linestring_generator()

        # Generate stream slope
        self.stream_slope_linestring_generator()

        # Write out stream bankfull width and discharge
        self.stream_bankfull_width_discharge_generator()

