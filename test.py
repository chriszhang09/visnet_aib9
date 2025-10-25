import argparse
import logging
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import pathlib
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch_geometric.data import Data

from aib9_lib import aib9_tools as aib9

from mse_training.visnet_vae_encoder_mse import ViSNetEncoderMSE
from mse_training.vae_model_mse import MolecularVAEMSE, vae_loss_function_mse
from vae_utils import validate_and_sample, visualize_molecule_3d, compute_bond_lengths

from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from pathlib import Path
import sys
import wandb
from torch.cuda.amp import autocast, GradScaler


# Training parameters
ATOM_COUNT = 58
COORD_DIM = 3
ORIGINAL_DIM = ATOM_COUNT * COORD_DIM  
LATENT_DIM = 128 
EPOCHS = 1
VISNET_HIDDEN_CHANNELS = 256
ENCODER_NUM_LAYERS = 3
DECODER_HIDDEN_DIM = 256
DECODER_NUM_LAYERS = 5
BATCH_SIZE = 1
LEARNING_RATE = 5e-5  
NUM_WORKERS = 2  # Parallel data loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


atom_feature_dim = 54 # Should be 10

visnet_params = {
    'hidden_channels': VISNET_HIDDEN_CHANNELS,
    'num_layers': ENCODER_NUM_LAYERS,
    'num_rbf': 32,
    'cutoff': 5.0,  # Used for cutoff-based edge identification
    'max_z': 3,
    'lmax': 2,  # Set to 1 for 3D vectors instead of 8D spherical harmonics
}

model = MolecularVAEMSE(
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
# 1, 58, 3 -> reshape to (58, 3) for PyG Data



ckpt = torch.load("checkpoints/vae_model_base_half_cv_105.pth", map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

in_vec = torch.randn(1, 58, 3, device=device)
in_vec = in_vec.squeeze(0)  # Remove batch dimension: (58, 3)

z = torch.tensor(ATOMIC_NUMBERS, dtype=torch.long, device=device)
data = Data(z=z, pos=in_vec)
data = data.to(device)
out_vec, _ = model.encoder(data)

# Create orthogonal matrix using PyTorch
A = torch.randn(3, 3, device=device)
Q, R = torch.linalg.qr(A)
orthogonal_matrix = Q
rot_in_vec = in_vec @ orthogonal_matrix
data = Data(z=z, pos=rot_in_vec)
data = data.to(device)
out_vec_2, _ = model.encoder(data)

print(torch.norm(out_vec_2 - R@out_vec).item())