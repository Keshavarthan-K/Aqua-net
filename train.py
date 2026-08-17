import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class AquaNet(torch.nn.Module):
    def __init__(self):
        super(AquaNet, self).__init__()
        self.conv1 = GCNConv(7, 16)
        self.conv2 = GCNConv(16, 8)
        # THE FIX 1: Remove the bias on the final layer so it can't hallucinate baseline water
        self.conv3 = GCNConv(8, 1, bias=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Static Scaling [Elev, Slope, Rain, Time, Drain, Depress, Road]
        scale_factors = torch.tensor([50.0, 0.5, 50.0, 24.0, 1.0, 5.0, 1.0], device=x.device)
        x_norm = x / scale_factors
        
        x = self.conv1(x_norm, edge_index)
        x = F.leaky_relu(x, negative_slope=0.1) 
        
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.1)
        
        x = self.conv3(x, edge_index)
        
        # THE FIX 2: Use ReLU instead of Softplus for a strict, hard 0.0 floor
        return torch.relu(x) 

if __name__ == "__main__":
    print("Loading the 7-Feature Spatial Dataset...")
    data = torch.load("chennai_mega_data.pt", weights_only=False)
    
    model = AquaNet()
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 
    
    # We repeat the Sunny Day twice in the curriculum to heavily enforce the zero-floor
    training_storms = [
        (40.0, 1.0),   # Flash Flood
        (20.0, 2.0),   # Heavy Rain
        (10.0, 5.0),   # Steady Rain
        (5.0, 24.0),   # Long Drizzle
        (30.0, 10.0),  # Monsoon Downpour
        (0.0, 1.0),    # Sunny Day
        (0.0, 24.0)    # Sunny Day (Long duration)
    ]
    
    # Pre-compute gravity flow directions
    elevations = data.x[:, 0]
    src, dst = data.edge_index
    flow_mask = elevations[src] > elevations[dst]
    flow_src = src[flow_mask]
    flow_dst = dst[flow_mask]
    
    print("\nTraining strict zero-boundary spatial AI...")
    for epoch in range(251):
        model.train()           
        optimizer.zero_grad()   
        epoch_loss = 0
        
        for sim_rain, sim_time in training_storms:
            data.x[:, 2] = sim_rain
            data.x[:, 3] = sim_time
            
            base_intensity = sim_rain / sim_time if sim_time > 0 else 0.0
            water_loads = torch.full((data.num_nodes,), base_intensity, dtype=torch.float32)
            
            # Spatial Gravity Simulation (Vectorized)
            runoff = water_loads[flow_src] * 0.20
            water_loads.scatter_add_(0, flow_src, -runoff)
            water_loads.scatter_add_(0, flow_dst, runoff)
            
            drainage_capacity = data.x[:, 4] * 3.5 
            excess = torch.clamp(water_loads - drainage_capacity, min=0.0)
            
            spatial_multiplier = 1.0 + (data.x[:, 5] * 0.5) 
            ground_truth_y = ((excess * sim_time) / 100.0) * spatial_multiplier
            
            data.y = ground_truth_y.unsqueeze(1)
            
            out = model(data)       
            loss = criterion(out, data.y) 
            loss.backward()         
            epoch_loss += loss.item()
            
        optimizer.step()        
        
        if epoch % 50 == 0:
            sample_pred = out[100].item()
            print(f"Epoch {epoch:03d} | Combined Loss: {epoch_loss:.4f} | Sample Node Prediction: {sample_pred:.3f} m")

    torch.save(model.state_dict(), "aquanet_brain.pth")
    print("\nTrue Spatial AI saved as 'aquanet_brain.pth'!")