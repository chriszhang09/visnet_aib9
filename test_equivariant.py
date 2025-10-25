import torch
import numpy as np
from torch_geometric.data import Data
from mse_training.vae_model_equivariant import EquivariantMolecularVAE, equivariant_vae_loss_function

# Test parameters
ATOM_COUNT = 58
LATENT_DIM = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Atomic numbers (same as in your original code)
ATOMIC_NUMBERS = [1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]

# Create equivariant model
model = EquivariantMolecularVAE(
    latent_dim=LATENT_DIM,
    num_atoms=ATOM_COUNT,
    atom_feature_dim=54,
    visnet_hidden_channels=256,
    decoder_hidden_dim=256,
    decoder_num_layers=5,
    visnet_kwargs={
        'hidden_channels': 256,
        'num_layers': 3,
        'num_rbf': 32,
        'cutoff': 5.0,
        'max_z': 54,
        'lmax': 1,  # Set to 1 for 3D vectors instead of 8D spherical harmonics
    }
).to(device)

print("Testing Equivariant VAE...")

# Create test molecule
in_vec = torch.randn(58, 3, device=device)
z = torch.tensor(ATOMIC_NUMBERS, dtype=torch.long, device=device)
data = Data(z=z, pos=in_vec)
data = data.to(device)

# Get latent representation
with torch.no_grad():
    mu, log_var = model.encoder(data)
    print(f"Original mu shape: {mu.shape}")
    print(f"Original log_var shape: {log_var.shape}")

# Create rotation matrix
A = torch.randn(3, 3, device=device)
Q, R = torch.linalg.qr(A)
orthogonal_matrix = Q

# Rotate input molecule
rot_in_vec = in_vec @ orthogonal_matrix
rot_data = Data(z=z, pos=rot_in_vec)
rot_data = rot_data.to(device)

# Get latent representation of rotated molecule
with torch.no_grad():
    mu_rot, log_var_rot = model.encoder(rot_data)
    print(f"Rotated mu shape: {mu_rot.shape}")
    print(f"Rotated log_var shape: {log_var_rot.shape}")

# Test equivariance: mu_rot should equal mu @ orthogonal_matrix
# (if the model is truly equivariant)
expected_mu_rot = mu @ orthogonal_matrix
expected_log_var_rot = log_var @ orthogonal_matrix

print(f"\nEquivariance Test:")
print(f"Expected mu_rot shape: {expected_mu_rot.shape}")
print(f"Actual mu_rot shape: {mu_rot.shape}")

# Compute differences
mu_diff = torch.norm(mu_rot - expected_mu_rot).item()
log_var_diff = torch.norm(log_var_rot - expected_log_var_rot).item()

print(f"Mu difference: {mu_diff:.6f}")
print(f"Log_var difference: {log_var_diff:.6f}")

if mu_diff < 1e-4 and log_var_diff < 1e-4:
    print("✅ Model is rotationally equivariant!")
else:
    print("❌ Model is NOT rotationally equivariant")
    print("The latent space does not preserve rotational structure")

# Test reconstruction
print(f"\nTesting reconstruction...")
with torch.no_grad():
    recon_data_list, mu_recon, log_var_recon = model(data)
    print(f"Reconstructed {len(recon_data_list)} molecules")
    print(f"Reconstruction shape: {recon_data_list[0].pos.shape}")
