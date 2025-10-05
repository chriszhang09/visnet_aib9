import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from pathlib import Path
import sys
import wandb
from vae_utils import visualize_molecule_3d, validate_and_sample

proj_path = Path().resolve()

from aib9_lib import aib9_tools as aib9
import os



# --- 1. Parameters ---
# These parameters define the model architecture and training process.
ATOM_COUNT = 58
COORD_DIM = 3
ORIGINAL_DIM = ATOM_COUNT * COORD_DIM  # 58 atoms * 3D coordinates = 174
LATENT_DIM = 4  # The dimension of the compressed representation.
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


data = np.load(aib9.FULL_DATA)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder Layers
        self.fc1 = nn.Linear(ORIGINAL_DIM, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, LATENT_DIM)
        self.fc_log_var = nn.Linear(128, LATENT_DIM)

        # Decoder Layers
        self.fc3 = nn.Linear(LATENT_DIM, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, ORIGINAL_DIM)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc_mean(h2), self.fc_log_var(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        return self.fc5(h4) # Linear activation for output coordinates

    def forward(self, x):
        # Flatten the input for the dense layers
        x_flat = x.view(-1, ORIGINAL_DIM)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        recon_flat = self.decode(z)
        # Reshape the output to match the original molecule structure
        recon = recon_flat.view(-1, ATOM_COUNT, COORD_DIM)
        return recon, mu, logvar

# --- 4. Loss Function ---
# The VAE loss is the sum of a reconstruction term and the KL divergence.
def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (Mean Squared Error)
    MSE = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return MSE + KLD

# --- 5. Training the VAE ---
def train():
    wandb.init(project="visnet_aib9")
    print("\n--- Generating Data ---")
    # Generate data and create a PyTorch DataLoader
    train_data_np = np.load(aib9.FULL_DATA)
    train_data_np = train_data_np.reshape(-1, 58, 3)
    train_data_tensor = torch.from_numpy(train_data_np)
    train_dataset = TensorDataset(train_data_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Generated training data with shape: {train_data_np.shape}")

    # Initialize model and optimizer
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f'Using device: {device}')
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- Starting Training ---")
    model.train() #
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            # data is a list, get the tensor
            molecules = data[0].to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(molecules)
            loss = loss_function(recon_batch, molecules, mu, logvar)

            wandb.log({"loss": loss.item()})
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_loss = train_loss / len(train_loader.dataset)
        print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
        torch.save(model.state_dict(), "model.pth")
    return model


def generate(model):
    print("\n--- Using the Trained Model for Generation ---")
    model.eval() # Set the model to evaluation mode
    
    with torch.no_grad():
        # Sample a random point from the latent space (standard normal distribution)
        device = next(model.parameters()).device
        random_latent_vector = torch.randn(1, LATENT_DIM).to(device)
        
        # Decode the latent vector to generate a new molecule
        generated_molecule_flat = model.decode(random_latent_vector)
        generated_molecule_coords = generated_molecule_flat.view(1, ATOM_COUNT, COORD_DIM)
        
        # Convert to numpy for inspection if needed
        validate_and_sample(model, generated_molecule_coords, device, [1, 6, 7, 8, 15, 16, 17], None, 0)
        visualize_molecule_3d(generated_molecule_coords, [1, 6, 7, 8, 15, 16, 17], "Generated Molecule")

        

if __name__ == "__main__":
    trained_model = train()
    generate(trained_model)
