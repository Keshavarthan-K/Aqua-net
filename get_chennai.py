import osmnx as ox

print("1. Initiating download for entire Chennai City road network...")
print("⚠️ WARNING: This is a massive dataset. It may take 5 to 15 minutes!")
print("Do not close the terminal. Please wait...")

# 1. Ask OpenStreetMap for the entire city boundaries
place_name = "Chennai, Tamil Nadu, India"

# 2. Download the graph (We turn off 'simplify' temporarily to speed up the raw download)
graph = ox.graph_from_place(place_name, network_type="drive", simplify=True)

print(f"\n2. Download Complete! The AI found {len(graph.nodes)} intersections in Chennai.")

print("3. Saving this massive graph to your hard drive...")
# We save it immediately so we never have to wait 15 minutes again!
ox.save_graphml(graph, "chennai_mega_network.graphml")

print("Success! 'chennai_mega_network.graphml' is saved and ready.")