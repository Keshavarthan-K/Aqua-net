import osmnx as ox
import torch
import numpy as np
from torch_geometric.data import Data

print("1. Loading the Chennai Mega-Map...")
try:
    graph = ox.load_graphml("chennai_3D_mega_network.graphml")
except Exception:
    graph = ox.load_graphml("chennai_mega_network.graphml")

print("2. Extracting elevations directly from local satellite data...")
try:
    graph = ox.elevation.add_node_elevations_raster(graph, "chennai_elevation.tif")
    print("Elevation extraction successful!")
except FileNotFoundError:
    print("⚠️ 'chennai_elevation.tif' not found. Place it in this folder and try again.")
    exit()

print("3. Calculating steepness (grade) of every road in Chennai...")
graph = ox.elevation.add_edge_grades(graph)

print("4. Mapping street intersections to integer IDs...")
node_mapping = {osmid: i for i, osmid in enumerate(graph.nodes())}

print("5. Calculating Drainage Index & Multi-Feature Node Matrix (X)...")
# Find min/max elevation across Chennai to normalize drainage scores
elevations = [float(graph.nodes[node].get('elevation', 0.0)) for node in graph.nodes()]
min_elev, max_elev = min(elevations), max(elevations)
elev_range = max_elev - min_elev if max_elev != min_elev else 1.0

# Calculate average slope (grade) connected to each intersection
node_slopes = {i: [] for i in range(len(graph.nodes()))}
for u, v, data in graph.edges(data=True):
    u_idx, v_idx = node_mapping[u], node_mapping[v]
    grade = abs(float(data.get('grade_abs', 0.0)))
    node_slopes[u_idx].append(grade)
    node_slopes[v_idx].append(grade)

node_features = []
labels = []

# We simulate storm parameters (e.g., 15 cm of rain falling over 3 hours)
# The GNN uses these to learn the relationship between intensity, time, and drainage!
simulated_rainfall_cm = 15.0  # Total rainfall in cm
simulated_duration_hr = 3.0   # Storm duration in hours

for osmid in graph.nodes():
    idx = node_mapping[osmid]
    elev = float(graph.nodes[osmid].get('elevation', 0.0))
    
    # Calculate slope score for this intersection
    slopes = node_slopes[idx]
    avg_slope = sum(slopes) / len(slopes) if len(slopes) > 0 else 0.0
    
    # DRAINAGE INDEX (0.05 to 1.0): 
    # High elevation & steep slope = High drainage (0.8 - 1.0)
    # Low elevation & flat terrain = Poor drainage (0.05 - 0.3)
    norm_elev = (elev - min_elev) / elev_range
    drainage_index = float(min(1.0, max(0.05, 0.7 * norm_elev + 0.3 * (avg_slope * 10))))
    
    # 4 Input Features: [Elevation, Rainfall (cm), Duration (hrs), Drainage Index]
    node_features.append([elev, simulated_rainfall_cm, simulated_duration_hr, drainage_index])
    
    # --- PHYSICAL GROUND TRUTH SIMULATION (Y) ---
    # Rainfall intensity in cm/hr
    intensity = simulated_rainfall_cm / simulated_duration_hr
    # Max drainage capacity of the street segment in cm/hr
    max_drainage_capacity = drainage_index * 3.5 
    
    # Accumulated water depth in meters
    excess_rate_cm_hr = max(0.0, intensity - max_drainage_capacity)
    accumulated_depth_m = (excess_rate_cm_hr * simulated_duration_hr) / 100.0
    
    labels.append([accumulated_depth_m])

x = torch.tensor(node_features, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.float)

print("6. Building Edge Index & Edge Attributes...")
source_nodes, target_nodes, edge_features = [], [], []
for u, v, data in graph.edges(data=True):
    source_nodes.append(node_mapping[u])
    target_nodes.append(node_mapping[v])
    length = float(data.get('length', 0.0))
    grade = float(data.get('grade_abs', 0.0))
    edge_features.append([length, grade])

edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
edge_attr = torch.tensor(edge_features, dtype=torch.float)

print("7. Compiling & Saving 'chennai_mega_data.pt'...")
chennai_mega_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
torch.save(chennai_mega_data, "chennai_mega_data.pt")

print("\n--- DATASET SPECS ---")
print(f"Total Intersections (Nodes): {chennai_mega_data.num_nodes}")
print(f"Total Road Segments (Edges): {chennai_mega_data.num_edges}")
print(f"Input Feature Dimensions per Node (X): {chennai_mega_data.x.shape[1]}")
print("Dataset successfully saved as 'chennai_mega_data.pt'!")