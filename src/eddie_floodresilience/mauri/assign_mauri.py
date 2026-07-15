import geopandas as gpd
import pandas as pd
from pathlib import Path

mauri_dir = Path(__file__).resolve().parent

mauri_df = pd.read_csv(mauri_dir / "mauri_table.csv")


mauri_df = mauri_df.drop(columns=["Maturity", "Future Mauri", "Future Maturity", "Target LCDB ?"])
mauri_df = mauri_df.drop_duplicates(ignore_index=True)
tangaroa_mauri = 5.0
tangaroa_classes = [20, 21, 22]
for tangaroa_class in tangaroa_classes:
    columns_to_change = ["Atua Domain", "Mauri"]
    query = mauri_df["LCDB class"] == tangaroa_class
    mauri_df.loc[query, columns_to_change] = ["Tangaroa (Sea, rivers, lakes)", tangaroa_mauri]

mauri_df["Mauri"] = mauri_df["Mauri"].fillna(1.0)

mauri_df.to_csv(mauri_dir / "simple_mauri_table.csv", index=False)


def join_with_mauri(landcover_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    joined = landcover_gdf.merge(mauri_df, "left", left_on="lcdb_class", right_on="LCDB class")
    joined = joined.drop(columns=["lcdb_class"])
    return joined

original_landcover_file = mauri_dir / "original_lcdb.geojson"
scenario_landcover_file = mauri_dir / "lcdb_scenario_6.geojson"

for file in [original_landcover_file, scenario_landcover_file]:
    landcover = gpd.read_file(file)
    joined_landcover = join_with_mauri(landcover)
    new_file_name = f"{file.stem}_joined{file.suffix}"
    joined_landcover.to_file(file.parent / new_file_name, driver="GeoJSON")

