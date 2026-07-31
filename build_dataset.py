import osmnx as ox
import torch
from torch_geometric.data import Data

print("1. Loading the saved map...")
graph = ox.load_graphml("tnagar_network.graphml")

print("2. Mapping street corners to math IDs...")
# PyTorch cannot read giant OSM ID numbers. 
# We must rename every intersection to a simple integer: 0, 1, 2, 3...
node_mapping = {osmid: i for i, osmid in enumerate(graph.nodes())}

print("3. Building 'x' -> Node Features (Elevations)...")
node_features = []
for node_id in graph.nodes():
    elevation = float(graph.nodes[node_id].get('elevation', 0.0))
    node_features.append([elevation])
x = torch.tensor(node_features, dtype=torch.float)

print("4. Building 'edge_index' and 'edge_attr' -> The Road Network...")
source_nodes = []
target_nodes = []
edge_features = []

for u, v, data in graph.edges(data=True):
    # Connect intersection U to intersection V
    source_nodes.append(node_mapping[u])
    target_nodes.append(node_mapping[v])
    
    # Get road length and steepness (grade)
    length = float(data.get('length', 0.0))
    grade = float(data.get('grade_abs', 0.0))
    edge_features.append([length, grade])

# PyTorch requires the connections to be 2 rows: [Source Nodes, Target Nodes]
edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
edge_attr = torch.tensor(edge_features, dtype=torch.float)

print("5. Generating 'y' -> Simulated Flood Labels...")
# For the prototype, we simulate historical data: areas below 12 meters are labeled '1' (Flooded).
# In a real deployment, you would swap this logic out with actual municipal flood datasets!
labels = []
for elevation in node_features:
    if elevation[0] < 12.0:
        labels.append([1.0]) # Flood Risk
    else:
        labels.append([0.0]) # Safe
y = torch.tensor(labels, dtype=torch.float)

print("6. Compiling the final PyTorch Geometric Data Object...")
# Combine everything into the ultimate 'Data' object
aqua_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# Save the compiled PyTorch dataset so our AI can load it instantly later
torch.save(aqua_data, "aqua_data.pt")

print("\n--- FINAL DATASET SPECS ---")
print(f"Number of Intersections (Nodes): {aqua_data.num_nodes}")
print(f"Number of Road Segments (Edges): {aqua_data.num_edges}")
print(f"Dataset successfully saved as 'aqua_data.pt'. Ready for Neural Network training!")