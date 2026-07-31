import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import osmnx as ox
import pandas as pd

# 1. Redefine the AI Architecture
class AquaNet(torch.nn.Module):
    def __init__(self):
        super(AquaNet, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 8)
        self.conv3 = GCNConv(8, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return torch.sigmoid(self.conv3(x, edge_index))

print("1. Waking up the trained AI Brain...")
model = AquaNet()
model.load_state_dict(torch.load("aquanet_brain.pth", weights_only=True))
model.eval()

print("2. AI is calculating flood vulnerabilities for all of Chennai...")
data = torch.load("chennai_mega_data.pt", weights_only=False)
with torch.no_grad():
    risk_scores = model(data).numpy()

print("3. Loading base map to extract GPS Coordinates...")
print("(This may take 30-60 seconds, but we only have to do it once!)")
graph = ox.load_graphml("chennai_mega_network.graphml")

print("4. Compiling the final web database...")
lats = []
lons = []

# Extract the exact latitude and longitude for every street corner
for node_id in graph.nodes():
    node_data = graph.nodes[node_id]
    lats.append(node_data['y'])
    lons.append(node_data['x'])

# Create a lightweight pandas database
df = pd.DataFrame({
    "lat": lats,
    "lon": lons,
    "elevation": data.x.numpy().flatten(),
    "ai_risk_score": risk_scores.flatten()
})

# Save it as a CSV file
df.to_csv("chennai_dashboard_data.csv", index=False)
print("✅ Success! 'chennai_dashboard_data.csv' is ready for the 3D App.")