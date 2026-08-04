import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class AquaNet(torch.nn.Module):
    def __init__(self):
        super(AquaNet, self).__init__()
        self.conv1 = GCNConv(4, 16)
        self.conv2 = GCNConv(16, 8)
        self.conv3 = GCNConv(8, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # 1. FEATURE SCALING: Normalizes inputs so the network doesn't choke on large numbers
        x_norm = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
        
        # 2. LEAKY RELU: Allows tiny negative numbers to pass, preventing "dead" gradients
        x = self.conv1(x_norm, edge_index)
        x = F.leaky_relu(x, negative_slope=0.1) 
        
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.1)
        
        x = self.conv3(x, edge_index)
        
        # 3. SOFTPLUS: A smooth curve that guarantees positive water depth without hitting 0
        return F.softplus(x) 

if __name__ == "__main__":
    print("Loading base dataset...")
    data = torch.load("chennai_mega_data.pt", weights_only=False)
    
    model = AquaNet()
    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 
    
    training_storms = [
        (40.0, 1.0),   # Flash Flood
        (20.0, 2.0),   # Heavy Rain
        (10.0, 5.0),   # Steady Rain
        (5.0, 24.0),   # Long Drizzle
        (30.0, 10.0)   # Monsoon Downpour
    ]
    
    print("\nTraining AI (Watch the Sample Node Depth rise!)...")
    for epoch in range(251):
        model.train()           
        optimizer.zero_grad()   
        epoch_loss = 0
        
        for sim_rain, sim_time in training_storms:
            data.x[:, 1] = sim_rain
            data.x[:, 2] = sim_time
            
            intensity = sim_rain / sim_time
            drainage_capacity = data.x[:, 3] * 3.5 
            excess = torch.clamp(intensity - drainage_capacity, min=0.0)
            
            ground_truth_y = (excess * sim_time) / 100.0
            data.y = ground_truth_y.unsqueeze(1)
            
            out = model(data)       
            loss = criterion(out, data.y) 
            loss.backward()         
            epoch_loss += loss.item()
            
        optimizer.step()        
        
        if epoch % 50 == 0:
            # We track a specific intersection to guarantee it learns to predict water
            sample_pred = out[100].item()
            print(f"Epoch {epoch:03d} | Combined Loss: {epoch_loss:.4f} | Sample Node Prediction: {sample_pred:.3f} m")

    torch.save(model.state_dict(), "aquanet_brain.pth")
    print("\nPhysics AI stabilized and saved as 'aquanet_brain.pth'!")