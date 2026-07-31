import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import osmnx as ox
import matplotlib.pyplot as plt

print("1. Rebuilding the AI Architecture...")
# We must define the exact same brain structure so PyTorch knows where to put the memories
class AquaNet(torch.nn.Module):
    def __init__(self):
        super(AquaNet, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 8)
        self.conv3 = GCNConv(8, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)

print("2. Waking up the trained AI...")
model = AquaNet()
# Load the memories you just trained!
model.load_state_dict(torch.load("aquanet_brain.pth", weights_only=True))
model.eval() # Tell the AI it is taking a test (no more studying/updating)

print("3. Loading the T. Nagar Map and Dataset...")
graph = ox.load_graphml("tnagar_network.graphml")
# Remember our fix from Bag 7!
data = torch.load("aqua_data.pt", weights_only=False) 

print("4. AI is analyzing the topography and predicting flood risks...")
with torch.no_grad(): # We turn off learning to save memory
    predictions = model(data)

print("5. Drawing the final AI Flood Map...")
# We will color the safe nodes Blue, and the flooded nodes Red
nodes = list(graph.nodes())
for i, node_id in enumerate(nodes):
    risk_score = predictions[i].item() # This is a number between 0 and 1
    
    # If the AI says there is more than a 50% chance of a flood:
    if risk_score > 0.50:
        graph.nodes[node_id]['color'] = '#ff0000' # Neon Red
    else:
        graph.nodes[node_id]['color'] = '#00bbff' # Cyan Blue

# Extract the colors in the exact right order for plotting
node_colors = [graph.nodes[n]['color'] for n in graph.nodes()]

print("Success! Check your screen for the AI's final assessment.")
# Plot the results
ox.plot_graph(graph, node_color=node_colors, node_size=12, edge_color="#333333", bgcolor="k")