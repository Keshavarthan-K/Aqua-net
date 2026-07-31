import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import time

# 1. Load the data we prepped in Bag #5
print("Loading the AquaNet Dataset...")
data = torch.load("chennai_mega_data.pt",weights_only=False)

# 2. Define the Neural Network Architecture
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
        return torch.sigmoid(x) # Squashes output between 0 (Safe) and 1 (Flooded)

# 3. Assemble the Brain
model = AquaNet()

# 4. Set up the Teacher (Loss Function) and the Student (Optimizer)
# BCELoss compares the AI's guesses to the 1s and 0s in our dataset
criterion = torch.nn.BCELoss() 
# Adam is the industry-standard algorithm that physically updates the AI's math
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

# 5. THE TRAINING LOOP
print("\nStarting AI Training for 200 Epochs...")
time.sleep(1) # Just a quick pause so you can read the terminal

for epoch in range(201):
    model.train()           # Tell the model it is in 'study mode'
    optimizer.zero_grad()   # Clear out the old math from the last guess
    
    out = model(data)       # The AI makes a guess for the whole city
    
    loss = criterion(out, data.y) # The Teacher grades the guess (lower loss is better!)
    loss.backward()         # The AI calculates how to fix its mistakes
    optimizer.step()        # The AI physically updates its brain cells
    
    # Print the progress every 20 epochs
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss (Error Rate): {loss.item():.4f}")

print("\nTraining Complete! The AI has learned the topography of T. Nagar.")

# 6. Save the trained brain!
torch.save(model.state_dict(), "aquanet_brain.pth")
print("Trained AI saved as 'aquanet_brain.pth'")