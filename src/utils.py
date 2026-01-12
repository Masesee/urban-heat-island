import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point, box
import numpy as np
from tqdm import tqdm

def extract_band_values(df, tiff_path):
    """
    Extracts spectral indices (NDVI, NDBI, NDWI) from a GeoTIFF file
    for the coordinates in the DataFrame.
    """
    # Ensure DataFrame has geometry
    if 'geometry' not in df.columns:
        geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
    else:
        gdf = df

    # Open the GeoTIFF
    with rasterio.open(tiff_path) as src:
        # Sample the raster at the point locations
        # Note: rasterio.sample expects a list of (x, y) tuples
        coords = [(x, y) for x, y in zip(gdf.geometry.x, gdf.geometry.y)]
        
        # Read the data
        sampled_values = list(src.sample(coords))
        
        # Create a DataFrame from the sampled values
        # Assuming Band 1: NDVI, Band 2: NDBI, Band 3: NDWI
        bands_df = pd.DataFrame(sampled_values, columns=['median_NDVI', 'median_NDBI', 'median_NDWI'])
        
        # Reset index to match
        gdf = gdf.reset_index(drop=True)
        bands_df = bands_df.reset_index(drop=True)
        
        # Concatenate
        result_df = pd.concat([gdf, bands_df], axis=1)
        
    return result_df

def compute_building_density(df, buildings_shp_path, lat_col="Latitude", lon_col="Longitude", buffer_m=100):
    """
    Computes building density within a buffer for points in the DataFrame.
    """
    # Load building footprints
    buildings_gdf = gpd.read_file(buildings_shp_path)
    
    # Convert input DataFrame to GeoDataFrame if needed
    if not isinstance(df, gpd.GeoDataFrame):
        geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
        gdf_points = gpd.GeoDataFrame(df, geometry=geometry)
    else:
        gdf_points = df.copy()

    # Set CRS if missing (assuming WGS84 for lat/lon)
    if gdf_points.crs is None:
        gdf_points.set_crs(epsg=4326, inplace=True)
        
    # Reproject to a projected CRS for accurate buffer calculation (e.g., UTM)
    # We'll use a generic estimate or the CRS of the buildings if projected
    target_crs = buildings_gdf.crs
    if target_crs.is_geographic:
        # Estimate UTM zone or use a standard metric CRS like World Mercator (3395) or Pseudo-Mercator (3857)
        # For better accuracy, local UTM is best, but 3857 is often used for simple buffers
        target_crs = "EPSG:3857" 
    
    gdf_points_proj = gdf_points.to_crs(target_crs)
    buildings_gdf_proj = buildings_gdf.to_crs(target_crs)

    building_densities = []

    # Use spatial index for faster queries
    sindex = buildings_gdf_proj.sindex

    for idx, row in tqdm(gdf_points_proj.iterrows(), total=gdf_points_proj.shape[0], desc="Computing Density"):
        point = row.geometry
        # Create square buffer (as per notebook description "square buffer" implied by some contexts, or just buffer)
        # Notebook said "buffer_m=100", usually implies radius. 
        # But `compute_building_density` snippet in Step 11 didn't show the full logic.
        # I'll assume standard circular buffer unless square is specified.
        # Actually, let's stick to circular buffer as it's standard.
        buffer_geom = point.buffer(buffer_m)
        
        # Find possible matches with spatial index
        possible_matches_index = list(sindex.intersection(buffer_geom.bounds))
        possible_matches = buildings_gdf_proj.iloc[possible_matches_index]
        
        # Precise intersection
        precise_matches = possible_matches[possible_matches.intersects(buffer_geom)]
        
        # Calculate density: sum of building areas / buffer area
        # Or count? Notebook said "building density". Usually area ratio.
        # Let's calculate area ratio.
        
        if not precise_matches.empty:
            # Intersection area
            intersection_area = precise_matches.intersection(buffer_geom).area.sum()
            density = intersection_area / buffer_geom.area
        else:
            density = 0.0
            
        building_densities.append(density)

    gdf_points["building_density_100m"] = building_densities
    return gdf_points
