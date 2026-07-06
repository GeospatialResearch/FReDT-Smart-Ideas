<?xml version="1.0" encoding="UTF-8"?>
<StyledLayerDescriptor xmlns="http://www.opengis.net/sld" version="1.0.0" xmlns:gml="http://www.opengis.net/gml"
                       xmlns:ogc="http://www.opengis.net/ogc" xmlns:sld="http://www.opengis.net/sld">
    <UserLayer>
        <sld:LayerFeatureConstraints>
            <sld:FeatureTypeConstraint/>
        </sld:LayerFeatureConstraints>
        <sld:UserStyle>
            <sld:Name>difference_r_b_0_2</sld:Name>
            <sld:FeatureTypeStyle>
                <sld:Rule>
                    <sld:RasterSymbolizer>
                        <sld:ChannelSelection>
                            <sld:GrayChannel>
                                <sld:SourceChannelName>1</sld:SourceChannelName>
                            </sld:GrayChannel>
                        </sld:ChannelSelection>
                        <sld:ColorMap type="ramp">
                            <sld:ColorMapEntry color="#070033" label="-0.2m" quantity="-0.2"/>
                            <sld:ColorMapEntry color="#0b11fb" label="-0.1m" quantity="-0.1"/>
                            <sld:ColorMapEntry color="#ffffff" label="-0.03m" quantity="-0.03" opacity="1"/>
                            <sld:ColorMapEntry color="#ffffff" quantity="-0.03" opacity="0"/>
                            <sld:ColorMapEntry color="#ffffff" quantity="0.03" opacity="0"/>
                            <sld:ColorMapEntry color="#ffffff" label="0.03m" quantity="0.03" opacity="1"/>
                            <sld:ColorMapEntry color="#fb110b" label="0.1m" quantity="0.1"/>
                            <sld:ColorMapEntry color="#350007" label="0.2m" quantity="0.2"/>
                        </sld:ColorMap>
                    </sld:RasterSymbolizer>
                </sld:Rule>
            </sld:FeatureTypeStyle>
        </sld:UserStyle>
    </UserLayer>
</StyledLayerDescriptor>
