<?xml version="1.0" encoding="UTF-8"?>
<StyledLayerDescriptor xmlns="http://www.opengis.net/sld" version="1.0.0" xmlns:gml="http://www.opengis.net/gml" xmlns:ogc="http://www.opengis.net/ogc" xmlns:sld="http://www.opengis.net/sld">
    <!--Style definition for Globcover Landcover rasters -->
    <UserLayer>
    <sld:LayerFeatureConstraints>
      <sld:FeatureTypeConstraint/>
    </sld:LayerFeatureConstraints>
    <sld:UserStyle>
      <sld:Name>wflow_landuse</sld:Name>
      <sld:FeatureTypeStyle>
        <sld:Rule>
          <sld:RasterSymbolizer>
            <sld:ChannelSelection>
              <sld:GrayChannel>
                <sld:SourceChannelName>1</sld:SourceChannelName>
              </sld:GrayChannel>
            </sld:ChannelSelection>
            <sld:ColorMap type="values">
              <sld:ColorMapEntry color="#007200" label="Evergreen Forest" quantity="40"/>
              <sld:ColorMapEntry color="#00441b" label="Dense Deciduous Forest" quantity="50"/>
              <sld:ColorMapEntry color="#38b000" label="Deciduous Forest" quantity="60"/>
              <sld:ColorMapEntry color="#66BB6A" label="Needleleaf Forest" quantity="70"/>
              <sld:ColorMapEntry color="#a5be00" label="Shrubland" quantity="130"/>
              <sld:ColorMapEntry color="#e69f00" label="Pasture Mosaic" quantity="120"/>
              <sld:ColorMapEntry color="#f0b84d" label="Grassland" quantity="140"/>
              <sld:ColorMapEntry color="#c7c7a6" label="Sparse Vegetation" quantity="150"/>
              <sld:ColorMapEntry color="#4a90e2" label="Wetland" quantity="160"/>
              <sld:ColorMapEntry color="#0000CD" label="Water" quantity="180"/>
              <sld:ColorMapEntry color="#A9A9A9" label="Bare Land" quantity="200"/>
            </sld:ColorMap>
          </sld:RasterSymbolizer>
        </sld:Rule>
      </sld:FeatureTypeStyle>
    </sld:UserStyle>
  </UserLayer>
</StyledLayerDescriptor>
