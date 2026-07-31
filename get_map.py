import osmnx as ox
import matplotlib.pyplot as plt

# 1. Download the base map of T. Nagar (X and Y coordinates)
address = "T. Nagar, Chennai, India"
print("Downloading street data...")
graph = ox.graph_from_address(address, dist=2000, network_type="drive")

# 2. Re-route OSMnx to use a FREE elevation database
# FIX: We now use {locations} instead of {} to match the new OSMnx update!
ox.settings.elevation_url_template = "https://api.opentopodata.org/v1/aster30m?locations={locations}"

# 3. Add elevation (Z coordinate) to every single street corner
# We use a batch size of 100 because the free API limits how much we can ask at once
print("Fetching elevation data for every street corner (this might take 60 seconds)...")
graph = ox.elevation.add_node_elevations_google(
    graph, api_key="dummy_key", batch_size=100, pause=0.5
)

# 4. Calculate the steepness (grade) of the actual roads connecting the corners
print("Calculating road steepness...")
graph = ox.elevation.add_edge_grades(graph)

# 5. Plot the map, but color the nodes based on their elevation!
print("Drawing elevation map...")
# Lower areas will be one color, higher areas will be another
nc = ox.plot.get_node_colors_by_attr(graph, "elevation", cmap="plasma")
ox.plot_graph(graph, node_color=nc, node_size=5, edge_color="#333333", bgcolor="k")
# 6. Save the graph to your computer so we don't have to download it again!
ox.save_graphml(graph, "tnagar_network.graphml")