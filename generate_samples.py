import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import sys
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
from aib9_lib import aib9_tools as aib9
import os
from tqdm import tqdm
import pathlib
from mse_training.vae_model_mse import MolecularVAEMSE
# --- 1. Parameters (Must match the trained model) ---
# It is crucial that these parameters are identical to the ones used
# when you trained and saved the model.
ATOM_COUNT = 58
COORD_DIM = 3
ORIGINAL_DIM = ATOM_COUNT * COORD_DIM  
LATENT_DIM = 128 
EPOCHS = 250
VISNET_HIDDEN_CHANNELS = 256
ENCODER_NUM_LAYERS = 3
DECODER_HIDDEN_DIM = 256
DECODER_NUM_LAYERS = 5
BATCH_SIZE = 128
LEARNING_RATE = 5e-5  
MODEL_PATH = 'checkpoints/vae_model_base_full_cv_250.pth' # The path to your saved model file

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

def h(x):
    """
    Implements the piecewise function h(x)
    h(x) = x if x < 1
    h(x) = 1 if x >= 1
    """
    if x < 1:
        return x
    else:
        return 1

def generate_latent_samples(num_samples=10000):
    
    
    samples = []
    total_attempts = 0
    
    M = 1.0

    while len(samples) < num_samples:
        total_attempts += 1
        z = np.random.normal(loc=0.0, scale=1.0, size=(LATENT_DIM,))
        
        z_matrix = z[:9].reshape((3, 3))
        det_z = np.linalg.det(z_matrix)
        h_det_z = h(det_z)
        prob_accept = np.exp(h_det_z - 1)
    
        u = np.random.uniform(0.0, 1.0)
        
        if u < prob_accept:
            # Accept the sample
            samples.append(z)
    
    print(total_attempts)
    return torch.tensor(samples,dtype=torch.float, device=device)
    

def generate_samples(model, num_samples=1):
    project_path = pathlib.Path(__file__).resolve().parent
    TOPO_FILE = (
        project_path / "aib9_lib/aib9_atom_info.npy"
    )  


    ATOMICNUMBER_MAPPING = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
    }

    ATOMIC_NUMBERS = []
    topo = np.load(TOPO_FILE)
    #print(f"Loaded topology with shape: {topo.shape}", topo)
    for i in range(topo.shape[0]):
        atom_name = topo[i, 0][0]
        if atom_name in ATOMICNUMBER_MAPPING:
            ATOMIC_NUMBERS.append(ATOMICNUMBER_MAPPING[atom_name])
        else:
            raise ValueError(f"Unknown atom name: {atom_name}")

    # Create all tensors directly on the selected device to avoid device mismatch
    z = torch.tensor(ATOMIC_NUMBERS, dtype=torch.long, device='cpu')


    # Create one-hot encoded atom types
    atom_types_one_hot = F.one_hot(z, num_classes=54).float().to(device)
    batch_size = 1000
    all_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    atomic_numbers = ATOMIC_NUMBERS

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating samples"):
            # Calculate actual batch size for this iteration
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            current_batch_size = end_idx - start_idx
            
            # Sample from prior
            z_samples = generate_latent_samples(current_batch_size)
            
            # Expand atom types and batch for this batch size
            atom_types_batch = atom_types_one_hot.unsqueeze(0).expand(current_batch_size, -1, -1)
            atom_types_batch = atom_types_batch.reshape(current_batch_size * len(atomic_numbers), -1)
            
            batch_batch = torch.arange(current_batch_size, device=device).repeat_interleave(len(atomic_numbers))
            
            # Generate coordinates
            generated_coords = model.decoder(z_samples, atom_types_batch, None, batch_batch)
            
            # Reshape to [batch_size, num_atoms, 3]
            generated_coords = generated_coords.reshape(current_batch_size, len(atomic_numbers), 3)
            
            # Move to CPU and convert to numpy
            generated_coords = generated_coords.cpu().numpy()
            all_samples.append(generated_coords)
            
            # Clear GPU memory
            del z_samples, atom_types_batch, batch_batch, generated_coords
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Concatenate all samples
    all_samples = np.concatenate(all_samples, axis=0)
    print(f"Generated {all_samples.shape[0]:,} samples with shape {all_samples.shape}")
    
    return all_samples

# --- Main Execution Block ---
if __name__ == "__main__":
    # Set the device to MPS (Apple Silicon GPU) if available, otherwise CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    atom_feature_dim = 54

    visnet_params = {
        'hidden_channels': VISNET_HIDDEN_CHANNELS,
        'num_layers': ENCODER_NUM_LAYERS,
        'num_rbf': 32,
        'cutoff': 5.0,  # Used for cutoff-based edge identification
        'max_z': 9,
    }

    loaded_model = MolecularVAEMSE(
        latent_dim=LATENT_DIM, 
        num_atoms=ATOM_COUNT, 
        atom_feature_dim=atom_feature_dim,
        visnet_hidden_channels=VISNET_HIDDEN_CHANNELS,
        decoder_hidden_dim=DECODER_HIDDEN_DIM,      
        decoder_num_layers=DECODER_NUM_LAYERS,         
        # No edge_index_template needed - PyG handles batching automatically
        visnet_kwargs=visnet_params,
        cutoff=5.0
    ).to(device)

    print(f"\nLoading trained weights from '{MODEL_PATH}'...")

    try:
        ckpt = torch.load(MODEL_PATH, map_location=device)
        loaded_model.load_state_dict(ckpt['model_state_dict'])
        loaded_model.eval()
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{MODEL_PATH}'.")
        print("Please make sure the 'model.pth' file is in the same directory as this script.")
        exit() # Exit the script if the model file is not found
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        exit()

    new_molecules = generate_samples(loaded_model, num_samples=100000)
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

    plt.savefig('pmf_new.png')

    try:
        np.save('molecule_coords_new.npy', new_molecules)
        print("\nSUCCESS: Coordinates saved to molecule_coords.npy")
    except Exception as e:
        print(f"\nAn error occurred while saving the coordinates: {e}")
