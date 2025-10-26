import sys
import os
# Add the parent directory to Python path to find visnet module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from torch_geometric.nn import global_add_pool
from visnet.models.visnet_block import ViSNetBlock
from visnet.models.output_modules import EquivariantVectorOutput, EquivariantEncoder, OutputModel, EquivariantVector
from torch_geometric.data import Data
from pytorch_lightning.utilities import rank_zero_warn
from torch.autograd import grad
from torch import Tensor
from torch_scatter import scatter
from visnet.models.output_modules import EquivariantVectorOutput
import pathlib
import numpy as np

class ViSNetDecoderMSE(nn.Module):
    """
    MSE-specific ViSNet encoder with proper initialization to prevent variance explosion.
    This version includes initialization fixes for stable MSE training.
    """
    
    def __init__(self, visnet_hidden_channels=128, prior_model=None, mean=None, std=None, **visnet_kwargs):
        """
        Args:
            latent_dim (int): The dimension of the latent space.
            visnet_hidden_channels (int): The hidden dimension for the ViSNet model.
            **visnet_kwargs: Additional arguments to pass to the ViSNet model constructor.
        """
        super().__init__()
        
        self.representation_model = ViSNetBlock(**visnet_kwargs)
        
        actual_hidden_channels = visnet_kwargs.get('hidden_channels', visnet_hidden_channels)
        
        self.output_model = EquivariantVector(hidden_channels=actual_hidden_channels)

        
    def forward(self, data):

        x, v = self.representation_model(data)
        v = self.output_model.pre_reduce(x, v)
        return v


if __name__ == "__main__":
    # Import the radius_graph function
    from torch_geometric.nn import radius_graph

    print("--- Running E(3) Equivariance Test for Encoder ---")

    # ... [All your setup code for ATOM_COUNT, LATENT_DIM, ATOMIC_NUMBERS, etc. is correct] ...
    
    project_path = pathlib.Path(__file__).resolve().parent.parent
    TOPO_FILE = (
        project_path / "aib9_lib/aib9_atom_info.npy"
    )  

    # Training parameters
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
    NUM_WORKERS = 2  # Parallel data loading


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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize the decoder
    visnet_params = {
        'hidden_channels': VISNET_HIDDEN_CHANNELS,
        'num_layers': ENCODER_NUM_LAYERS,
        'num_rbf': 32,
        'cutoff': 5.0,  # Used for cutoff-based edge identification
        'max_z': max(ATOMIC_NUMBERS) + 1,
        'lmax':1
    }
    pooling_model = EquivariantVector(hidden_channels=VISNET_HIDDEN_CHANNELS)
    pos_rand = torch.randn(ATOM_COUNT, 3, 256).to(device).float()
    print(pos_rand.shape)
    # 3. Create the Data object using the pre-computed edge_index
    x = torch.randn(ATOM_COUNT, 256).to(device).float()
    v = pooling_model.pre_reduce(x, pos_rand)
    
    v= v.squeeze(-1)
    print(v.shape)
    # Create fully connected edge_index
    edge_index = torch.combinations(torch.arange(ATOM_COUNT, device=device), 2).t().contiguous()
    # Add both directions for undirected graph
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    data = Data(z=z, pos=v, edge_index=edge_index)

    # 4. Generate a Random 3D Rotation Matrix (R)
    A = torch.randn(3, 3).to(device).float()
    R, _ = torch.linalg.qr(A)
    if torch.det(R) < 0:
        R[:, 0] *= -1
    pos_rand = pos_rand.transpose(1, 2)
    pos_rand_rotated = torch.matmul(pos_rand, R).float().transpose(1, 2)
    w = pooling_model.pre_reduce(x, pos_rand_rotated)
    w = w.squeeze(-1)
    data_rotated = Data(z=z, pos=w, edge_index=edge_index)
    
    # --- END OF FIX ---

    decoder= ViSNetDecoderMSE(
        visnet_hidden_channels=VISNET_HIDDEN_CHANNELS,
        **visnet_params
    ).to(device).float()
    decoder.eval()

    print("\nGenerated random 3x3 rotation matrix R.")

    # 3. Path 1: Encode original coordinates
    with torch.no_grad():
        X_out_1 = decoder(data)
    
    print("Path 1 (Encode original) complete.")

    # 4. Path 2: Encode rotated coordinates  
    with torch.no_grad():
        X_out_2 = decoder(data_rotated)
    print("Path 2 (Encode rotated) complete.")
    
    # --- TEST LOGIC ---
    # Your test logic is correct for EQUIVARIANCE, not invariance.
    # We are testing if f(R(x)) == R(f(x))
    
    # Squeeze [58, 3, 1] to [58, 3]
    X_out_1 = X_out_1.transpose(1, 2)
    X_out_2 = X_out_2.transpose(1, 2)
    
    # Manually rotate the original output: R(f(x))
    # Note: We are doing (f(x) @ R) which is correct for [N, 3] row vectors
    X_out_1_rotated = torch.matmul(X_out_1, R)
    
    # Compare: norm( R(f(x)) - f(R(x)) )
    mu_diff = torch.norm(X_out_1_rotated - X_out_2)
    print(X_out_1_rotated - X_out_2)
  
    print(f"\nEquivariance difference: {mu_diff.item():.6f}")

    tolerance = 1e-4 # You may need to relax this slightly (e.g., 1e-5) for float32
    is_equivariant = (mu_diff.item() < tolerance)

    print("\n--- TEST RESULTS ---")
    print(f"Encoder's vector output is rotation-equivariant: {is_equivariant}")

    if is_equivariant:
        print("✅ E(3) Equivariance Test PASSED!")
        print("The encoder's vector output f(v) correctly transforms as f(R(x)) = R(f(x)).")
    else:
        print("❌ E(3) Equivariance Test FAILED!")
        print(f"Expected differences < {tolerance}, got: {mu_diff.item():.6f}")