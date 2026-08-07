import osmnx as ox
import torch
import numpy as np
from torch_geometric.data import Data

print("1. Loading the Chennai Mega-Map...")
graph_file = "chennai_mega_network.graphml"
graph = ox.load_graphml(graph_file)

print("2. Calculating Node Geometry (Slope & Depression)...")
node_mapping = {osmid: i for i, osmid in enumerate(graph.nodes())}
node_slopes = {i: [] for i in range(len(graph.nodes()))}
node_elevations = {osmid: float(data.get('elevation', 0.0)) for osmid, data in graph.nodes(data=True)}

# Simple encoding for road types (higher number = more paved/less drainage)
road_encodings = {'motorway': 1.0, 'trunk': 0.9, 'primary': 0.8, 'secondary': 0.7, 'tertiary': 0.6, 'residential': 0.4, 'unclassified': 0.3}

for u, v, data in graph.edges(data=True):
    u_idx, v_idx = node_mapping[u], node_mapping[v]
    grade = abs(float(data.get('grade_abs', 0.0)))
    node_slopes[u_idx].append(grade)
    node_slopes[v_idx].append(grade)

node_features = []
labels = []

# Base simulated storm for the dataset generation
simulated_rainfall_cm = 15.0  
simulated_duration_hr = 3.0   
base_intensity = simulated_rainfall_cm / simulated_duration_hr

# Step A: Calculate Spatial Runoff (Gravity Simulation)
print("3. Simulating Gravity-Driven Water Flow (Runoff)...")
water_loads = {osmid: base_intensity for osmid in graph.nodes()}

for u, v in graph.edges():
    elev_u = node_elevations[u]
    elev_v = node_elevations[v]
    
    # If u is higher than v, 20% of u's water flows down to v
    if elev_u > elev_v:
        runoff = water_loads[u] * 0.20
        water_loads[u] -= runoff
        water_loads[v] += runoff

print("4. Compiling the 7-Feature Matrix...")
for osmid, data in graph.nodes(data=True):
    idx = node_mapping[osmid]
    elev = node_elevations[osmid]
    
    # 1. Local Slope
    slopes = node_slopes[idx]
    avg_slope = sum(slopes) / len(slopes) if slopes else 0.0
    
    # 2. Depression Depth (How much of a "bowl" is this intersection?)
    neighbors = list(graph.successors(osmid)) + list(graph.predecessors(osmid))
    if neighbors:
        min_neighbor_elev = min([node_elevations[n] for n in neighbors])
        # If neighbors are higher, this node is in a bowl
        depression_depth = max(0.0, min_neighbor_elev - elev) 
    else:
        depression_depth = 0.0
        
    # 3. Road Type Encoding
    highway_tag = data.get('highway', 'unclassified')
    # Handle cases where highway tag is a list
    if isinstance(highway_tag, list):
        highway_tag = highway_tag[0]
    road_type_val = road_encodings.get(highway_tag, 0.5)
    
    # 4. Drainage Index (Placeholder formula)
    drainage_index = min(1.0, max(0.05, 0.3 + (avg_slope * 10) - (road_type_val * 0.2)))
    
    # 5. Assemble the 7 columns in exact order!
    # [elevation, local_slope, rainfall_cm, duration_hr, drain_index, depression_depth, road_type_encoded]
    feature_row = [
        elev, 
        avg_slope, 
        simulated_rainfall_cm, 
        simulated_duration_hr, 
        drainage_index, 
        depression_depth, 
        road_type_val
    ]
    node_features.append(feature_row)
    
    # 6. Calculate the True Spatial Label (Y) using the runoff we simulated earlier
    effective_intensity = water_loads[osmid]
    max_drainage_capacity = drainage_index * 3.5 
    
    excess_rate = max(0.0, effective_intensity - max_drainage_capacity)
    accumulated_depth_m = (excess_rate * simulated_duration_hr) / 100.0
    labels.append([accumulated_depth_m])

x = torch.tensor(node_features, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.float)

print("5. Building Edge Index...")
source_nodes, target_nodes = [], []
for u, v in graph.edges():
    source_nodes.append(node_mapping[u])
    target_nodes.append(node_mapping[v])

edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

print("6. Saving 'chennai_mega_data.pt'...")
chennai_mega_data = Data(x=x, edge_index=edge_index, y=y)
torch.save(chennai_mega_data, "chennai_mega_data.pt")
print(f"Success! Matrix shape: {chennai_mega_data.x.shape} (Should be [Nodes, 7])")