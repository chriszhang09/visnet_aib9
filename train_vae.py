import argparse
import logging
import os
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import pathlib


from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch_geometric.data import Data

from aib9_lib import aib9_tools as aib9
from vae_decoder import EGNNDecoder
from vae_model import MolecularVAE, vae_loss_function
from visnet_vae_encoder import ViSNetEncoder

from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from pathlib import Path
import sys
import wandb
from aib9_lib import aib9_tools as aib9


def main():

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'Using device: {device}')


    ATOM_COUNT = 58
    COORD_DIM = 3
    ORIGINAL_DIM = ATOM_COUNT * COORD_DIM  
    LATENT_DIM = 30 
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3

    train_data_np = np.load(aib9.FULL_DATA)
    train_data_np = train_data_np.reshape(-1, 58, 3)

    
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

    z = torch.tensor(ATOMIC_NUMBERS, dtype=torch.long)

    # Create a list of Data objects, one for each molecule


    
    edges = aib9.identify_all_covalent_edges(topo)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    train_data_list = []
    for i in range(train_data_np.shape[0]):
        pos = torch.from_numpy(train_data_np[i]).float()
        # Add edge_index to each data object
        data = Data(z=z, pos=pos, edge_index=edge_index) 
        train_data_list.append(data)
    train_loader = DataLoader(train_data_list, batch_size=BATCH_SIZE, shuffle=True)
    model = MolecularVAE(visnet_model=ViSNetEncoder, latent_dim=LATENT_DIM, num_atoms=ATOM_COUNT, atom_feature_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE )
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            molecules = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(molecules)
            loss = vae_loss_function(recon_batch, molecules.pos, mu, logvar)

            # wandb.log({"loss": loss.item()}) # Uncomment if you are using wandb
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_loss = train_loss / len(train_loader.dataset)
        print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')

if __name__ == "__main__":
    main()
