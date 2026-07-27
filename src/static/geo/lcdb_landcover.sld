<?xml version="1.0" encoding="UTF-8"?>
<StyledLayerDescriptor xmlns="http://www.opengis.net/sld" version="1.0.0" xmlns:gml="http://www.opengis.net/gml"
                       xmlns:ogc="http://www.opengis.net/ogc" xmlns:sld="http://www.opengis.net/sld">
    <!--Style definition for Globcover Landcover rasters -->
    <UserLayer>
        <sld:LayerFeatureConstraints>
            <sld:FeatureTypeConstraint/>
        </sld:LayerFeatureConstraints>
        <sld:UserStyle>
            <sld:Name>lcdb_landcover</sld:Name>
            <sld:FeatureTypeStyle>
                <sld:Rule>
                    <sld:RasterSymbolizer>
                        <sld:ChannelSelection>
                            <sld:GrayChannel>
                                <sld:SourceChannelName>1</sld:SourceChannelName>
                            </sld:GrayChannel>
                        </sld:ChannelSelection>
                        <sld:ColorMap type="values">
                            <sld:ColorMapEntry color="#e69f00" label="High producing Exotic Grassland" quantity="40"/>
                            <sld:ColorMapEntry color="#f0b84d" label="Low Producing Grassland" quantity="41"/>
                            <sld:ColorMapEntry color="#4a90e2" label="Herbaceous Freshwater Vegetation" quantity="45"/>
                            <sld:ColorMapEntry color="#a5be00" label="Manuka and/or Kanuka" quantity="52"/>
                            <sld:ColorMapEntry color="#38b000" label="Broadleaved Indigenous Hardwoods" quantity="54"/>
                            <sld:ColorMapEntry color="#c7c7a6" label="Forest - Harvested" quantity="64"/>
                            <sld:ColorMapEntry color="#00441b" label="Deciduous Hardwoods" quantity="68"/>
                            <sld:ColorMapEntry color="#007200" label="Indigenous Forest" quantity="69"/>
                            <sld:ColorMapEntry color="#66BB6A" label="Exotic Forest (needleleaf forest)" quantity="71"/>
                        </sld:ColorMap>
                    </sld:RasterSymbolizer>
                </sld:Rule>
            </sld:FeatureTypeStyle>
        </sld:UserStyle>
    </UserLayer>
</StyledLayerDescriptor>
