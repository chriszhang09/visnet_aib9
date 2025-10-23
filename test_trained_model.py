#!/usr/bin/env python3
"""
Test reconstruction loss on test set and generate PMF plots for reconstructed molecules.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_cluster import radius_graph
from sklearn.model_selection import train_test_split
import argparse
import pathlib
from tqdm import tqdm
import wandb

ATOM_COUNT = 58
COORD_DIM = 3
ORIGINAL_DIM = ATOM_COUNT * COORD_DIM  
LATENT_DIM = 128 
EPOCHS = 250
VISNET_HIDDEN_CHANNELS = 256
ENCODER_NUM_LAYERS = 3
DECODER_HIDDEN_DIM = 256
DECODER_NUM_LAYERS = 5
BATCH_SIZE = 256
LEARNING_RATE = 5e-5  
MODEL_PATH = 'checkpoints/vae_model_base_full_cv_250.pth'


# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from aib9_lib import aib9_tools as aib9
from mse_training.vae_model_mse import MolecularVAEMSE

def pairwise_distance_loss(true_coords, pred_coords, p=2):
    device = torch.device('cuda')
    if pred_coords.device != true_coords.device:
        pred_coords = pred_coords.to(device)
        true_coords = true_coords.to(device)
    true_distances = torch.pdist(true_coords, p=p)
    pred_distances = torch.pdist(pred_coords, p=p)
    loss = F.mse_loss(pred_distances, true_distances)
    return loss

def evaluate_reconstruction(model, test_loader, device, num_samples=1000):
    """Evaluate reconstruction loss on test set."""
    print(f"Evaluating reconstruction on {num_samples} test samples...")
    
    model.eval()
    total_recon_loss = 0.0
    num_batches = 0
    
    all_reconstructed = []
    all_original = []
    all_latent_mu = []
    all_latent_log_var = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader, desc="Evaluating")):
            if batch_idx * test_loader.batch_size >= num_samples:
                break
                
            data = data.to(device)
            
            # Forward pass
            reconstructed_pos, mu, log_var = model(data)
            
            # Compute loss
            recon_loss = pairwise_distance_loss(reconstructed_pos, data.pos, 2)
            
            total_recon_loss += recon_loss.item() 
            print(f"Reconstruction Loss: {recon_loss.item()}")
            num_batches += 1
            batch_size = data.batch.max().item() + 1
            reconstructed_reshaped = reconstructed_pos.cpu().numpy().reshape(batch_size, 58, 3)
            all_reconstructed.append(reconstructed_reshaped)
    
    # Average losses
    avg_recon_loss = total_recon_loss / num_batches
    
    print(f"Average Reconstruction Loss: {avg_recon_loss:.6f}")

    return {

        'avg_recon_loss': avg_recon_loss,
        'reconstructed': np.concatenate(all_reconstructed, axis=0),
    }


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    print(f"Using device: {device}")
    

    atom_feature_dim = 54

    visnet_params = {
        'hidden_channels': VISNET_HIDDEN_CHANNELS,
        'num_layers': ENCODER_NUM_LAYERS,
        'num_rbf': 32,
        'cutoff': 5.0,  # Used for cutoff-based edge identification
        'max_z': 9,
    }

    model = MolecularVAEMSE(
        latent_dim=LATENT_DIM, 
        num_atoms=ATOM_COUNT, 
        atom_feature_dim=atom_feature_dim,
        visnet_hidden_channels=VISNET_HIDDEN_CHANNELS,
        decoder_hidden_dim=DECODER_HIDDEN_DIM,      
        decoder_num_layers=DECODER_NUM_LAYERS,         
        visnet_kwargs=visnet_params,
        cutoff=5.0
    ).to(device)

    print(f"\nLoading trained weights from '{MODEL_PATH}'...")

    try:
        ckpt = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print("Model weights loaded successfully.")
        model.eval()
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        exit() 

 
    train_data_np = np.load(aib9.FULL_DATA)
    data = np.load(aib9.FULL_DATA).reshape(-1, 58, 3)
    cv = aib9.kai_calculator(data)
    mask = cv[:, 0] > 0
    filtered_data = data[mask]
    filtered_cv = cv[mask]

    L=np.array([
        [-1,0,0],
        [0,1,0],
        [0,0,1]
    ])

    print("original cv:\n", cv)
    
    train_data, test_data = train_test_split(filtered_data, test_size=0.1, random_state=42)
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

    # Create a list of Data objects, one for each molecule
    # Use cutoff-based edge identification instead of predefined covalent edges
    test_data_objects = []
    for i in range(test_data.shape[0]):
        pos = torch.from_numpy(test_data[i]).float().to('cpu')
        # No edge_index - let ViSNet use cutoff-based edge identification
        data = Data(z=z, pos=pos)
        test_data_objects.append(data)

    test_loader = DataLoader(
        test_data_objects,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    # Evaluate reconstruction
    results = evaluate_reconstruction(model, test_loader, device, test_data.shape[0])
    
    # Save results
    print("Saving results...")
    np.save('reconstructed_coords.npy', results['reconstructed'])
    
    kai = aib9.kai_calculator(results['reconstructed'])
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
    plt.savefig('pmf_reconstruction_large_data.png')


if __name__ == "__main__":
    main()

