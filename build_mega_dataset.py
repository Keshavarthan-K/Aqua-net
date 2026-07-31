import osmnx as ox
import rasterio

print("1. Loading the Chennai Mega-Map (68,664 nodes)...")
graph = ox.load_graphml("chennai_mega_network.graphml")

print("2. Extracting elevations directly from local satellite data...")
# This built-in OSMnx function acts like a cookie-cutter. 
# It stamps our street graph onto the satellite image and extracts the height for every node instantly.
try:
    graph = ox.elevation.add_node_elevations_raster(graph, "chennai_elevation.tif")
    print("Elevation extraction successful!")
except FileNotFoundError:
    print("⚠️ 'chennai_elevation.tif' not found. You need to download a DEM file for Chennai!")
    print("Once downloaded, place it in this folder and run again.")
    exit()

print("3. Calculating the steepness (grade) of every road in the city...")
# This calculates the slopes so the AI knows which way the water will slide
graph = ox.elevation.add_edge_grades(graph)

print("4. Saving the finalized 3D Mega-Map...")
ox.save_graphml(graph, "chennai_3D_mega_network.graphml")

print("Success! Your city-scale dataset is ready for the Graph Neural Network.")