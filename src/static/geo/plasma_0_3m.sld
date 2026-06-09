<?xml version="1.0" encoding="UTF-8"?>
<StyledLayerDescriptor xmlns="http://www.opengis.net/sld" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                       xsi:schemaLocation="http://www.opengis.net/sld
http://schemas.opengis.net/sld/1.0.0/StyledLayerDescriptor.xsd" version="1.0.0">
    <NamedLayer>
        <Name>plasma_0_3m</Name>
        <UserStyle>
            <Title>plasma_0_3m</Title>
            <FeatureTypeStyle>
                <Rule>
                    <RasterSymbolizer>
                        <ColorMap type="ramp">
                            <ColorMapEntry color="#0d0887" label="0m" quantity="0" opacity="0"/>
                            <ColorMapEntry color="#0d0887" quantity="0.03" opacity="0"/>
                            <ColorMapEntry color="#0d0887" label="0.03m" quantity="0.03" opacity="1"/>
                            <ColorMapEntry color="#6b00a8" label="0.1m" quantity="0.1"/>
                            <ColorMapEntry color="#b02a90" label="0.3m" quantity="0.3"/>
                            <ColorMapEntry color="#e06462" label="0.8m" quantity="0.8"/>
                            <ColorMapEntry color="#fca736" label="1.6m" quantity="1.6"/>
                            <ColorMapEntry color="#f0f921" label="3m" quantity="3"/>
                        </ColorMap>
                    </RasterSymbolizer>
                </Rule>
            </FeatureTypeStyle>
        </UserStyle>
    </NamedLayer>
</StyledLayerDescriptor>
