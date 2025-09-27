import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from pathlib import Path
import sys
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
from aib9_lib import aib9_tools as aib9
import os

# --- 1. Parameters (Must match the trained model) ---
# It is crucial that these parameters are identical to the ones used
# when you trained and saved the model.
ATOM_COUNT = 58
COORD_DIM = 3
ORIGINAL_DIM = ATOM_COUNT * COORD_DIM
LATENT_DIM = 4
MODEL_PATH = 'model.pth' # The path to your saved model file

# --- 2. VAE Model Definition (Must match the trained model) ---
# This class defines the neural network architecture. You must use the
# exact same structure as the model you saved, otherwise the saved
# weights (the "state dictionary") will not match the layers.
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder Layers (Not used for generation, but needed to define the model)
        self.fc1 = nn.Linear(ORIGINAL_DIM, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, LATENT_DIM)
        self.fc_log_var = nn.Linear(128, LATENT_DIM)

        # Decoder Layers (This is the part we will use)
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
        # Use linear activation for output coordinates
        return self.fc5(h4)

    def forward(self, x):
        x_flat = x.view(-1, ORIGINAL_DIM)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        recon_flat = self.decode(z)
        recon = recon_flat.view(-1, ATOM_COUNT, COORD_DIM)
        return recon, mu, logvar

# --- 3. Generation Function ---
def generate_samples(model, num_samples=1):
    """
    Generates new samples using the decoder part of the trained VAE.

    Args:
        model (VAE): The loaded VAE model with trained weights.
        num_samples (int): The number of new molecular samples to generate.

    Returns:
        numpy.ndarray: An array of generated molecular coordinates.
    """
    print(f"\n--- Generating {num_samples} new sample(s) ---")
    
    # Set the model to evaluation mode. This disables layers like dropout
    # that behave differently during training vs. inference.
    model.eval()

    # We don't need to calculate gradients for generation, which saves memory and computation
    with torch.no_grad():
        # Determine the device the model is on (CPU or MPS)
        device = next(model.parameters()).device
        
        # Sample random points from the latent space. The VAE was trained to make
        # this space resemble a standard normal distribution (mean=0, variance=1).
        random_latent_vectors = torch.randn(num_samples, LATENT_DIM).to(device)
        
        # Use the decoder to transform the latent vectors back into molecular coordinates
        generated_samples_flat = model.decode(random_latent_vectors)
        
        # Reshape the flat output back to the 58x3 coordinate structure
        generated_samples_coords = generated_samples_flat.view(num_samples, ATOM_COUNT, COORD_DIM)
        
        # Move the tensor to the CPU and convert it to a NumPy array for easier use
        generated_samples_np = generated_samples_coords.cpu().numpy()
        
        print(f"Successfully generated samples with final shape: {generated_samples_np.shape}")
        return generated_samples_np

# --- Main Execution Block ---
if __name__ == "__main__":
    # Set the device to MPS (Apple Silicon GPU) if available, otherwise CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')


    loaded_model = VAE().to(device)
    print(f"\nLoading trained weights from '{MODEL_PATH}'...")

    try:
        loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{MODEL_PATH}'.")
        print("Please make sure the 'model.pth' file is in the same directory as this script.")
        exit() # Exit the script if the model file is not found
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        exit()

    new_molecules = generate_samples(loaded_model, num_samples=1000000)
    kai = aib9.kai_calculator(new_molecules)
    nbins = 200
    kai_flat = kai.reshape(-1, 2)
    H, xedges, yedges, binnumber = binned_statistic_2d(kai_flat[:,0], kai_flat[:,1], None, statistic='count', bins=nbins)
    H = H.T  # Transpose so that the orientation is correct
    H = H / np.sum(H)  # Normalize to get a probability distribution
    pmf = np.full_like(H, np.nan)
    mask = H > 0
    pmf[mask] = -np.log(H[mask])
    pmf_min = np.nanmin(pmf)
    pmf = pmf - pmf_min  
    plt.figure(figsize=(6,5))
    plt.imshow(pmf, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='viridis', interpolation='nearest', vmin=0, vmax=8)
    plt.colorbar(label='PMF (kT)')
    plt.xlabel('kai1')
    plt.ylabel('kai2')

    plt.savefig('pmf.png')
    coords =new_molecules[0]
    view   =aib9.py3dplot(coords)
