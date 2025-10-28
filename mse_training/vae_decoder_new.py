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

