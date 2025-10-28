from tkinter import S
import torch
import torch.nn as nn
import torch.nn.functional as F
from .visnet_vae_encoder_mse import ViSNetEncoderMSE
from .vae_decoder_new import ViSNetDecoderMSE
from visnet.models.output_modules import EquivariantVectorOutput, EquivariantEncoder, OutputModel, EquivariantVector
from torch_geometric.data import Data
class MolecularVAEMSE(nn.Module):
    """
    MSE-specific Molecular VAE with proper initialization for stable training.
    """
    def __init__(self, latent_dim, num_atoms, atom_feature_dim, 
                 visnet_hidden_channels=128, decoder_hidden_dim=128, 
                 decoder_num_layers=6, visnet_kwargs=None, cutoff: float = 3.0):
        """
        Args:
            latent_dim (int): Dimension of the latent space
            num_atoms (int): Number of atoms in each molecule
            atom_feature_dim (int): Dimension for one-hot encoding of atom types (max_atomic_number + 1)
            visnet_hidden_channels (int): Hidden dimension for ViSNet encoder
            decoder_hidden_dim (int): Hidden dimension for EGNN decoder
            decoder_num_layers (int): Number of EGNN layers in decoder
            visnet_kwargs (dict): Additional parameters for ViSNetBlock
        """
        super().__init__()
        if visnet_kwargs is None:
            visnet_kwargs = {}
        
        self.encoder = ViSNetEncoderMSE(
            latent_dim=latent_dim,
            visnet_hidden_channels=visnet_hidden_channels,
            **visnet_kwargs
        )
        self.decoder = ViSNetDecoderMSE(
            visnet_hidden_channels=decoder_hidden_dim,
            **visnet_kwargs
        )
        self.latent_dim = latent_dim
        self.atom_feature_dim = atom_feature_dim
        self.num_atoms = num_atoms
        # Use the actual hidden channels from the encoder output
        # The encoder output will have visnet_hidden_channels, so use that
        self.pooling_model = EquivariantVector(hidden_channels=latent_dim)
        self.linear_layer = nn.Linear(visnet_hidden_channels, visnet_hidden_channels)
        self.activation = nn.ReLU()
        self.linear_layer2 = nn.Linear(visnet_hidden_channels, latent_dim)
        # Check for NaN in model parameters after initialization
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f"Warning: NaN detected in model parameter {name} after initialization")
                with torch.no_grad():
                    param.data = torch.randn_like(param.data) * 0.01

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick for non-centered isotropic Gaussian.
        z ~ N(μ, σ²I) where σ² is a scalar per-sample.
        Handles log_var shapes [batch], [batch, 1], or scalar.
        Uses soft bounds to prevent numerical issues while allowing learning.
        """
        # Expand log_var to match mu's shape [58, 3, N] -> [58, 3, N]
        # log_var is [58, N], we need [58, 3, N]
        
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu)
        return mu + eps * std
    
    def sample_prior(self, device):
        """Sample from 3D isotropic Gaussian prior N(0, I) with shape (num_atoms, H, 3)"""
        return torch.randn(self.num_atoms, 3, self.latent_dim, device=device)

    def forward(self, data):

        x, v = self.encoder(data)

        log_var = x[:, :self.latent_dim]
        log_var_expanded = log_var.unsqueeze(1).expand(-1, 3, -1)
        mu = v

        latent_vector = self.reparameterize(mu, log_var_expanded).float()

        latent_vector = self.pooling_model.pre_reduce(torch.ones(latent_vector.shape[0], self.latent_dim, device=data.pos.device), latent_vector).squeeze(-1)

        num_nodes = data.num_nodes

        
        # Create batch tensor for the edge_index
        batch = torch.zeros(num_nodes, dtype=torch.long, device=data.pos.device)
        edge_index = torch.combinations(torch.arange(num_nodes, device=data.pos.device), 2).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        data_decoder = Data(z=data.z, pos=latent_vector, edge_index=data.edge_index, batch=batch)

        reconstructed_pos = self.decoder(data_decoder).squeeze(-1)
        return reconstructed_pos, mu, log_var_expanded
    
    def reset_parameters(self):
        """Reset all parameters in the model"""
        # Reset encoder and decoder parameters
        if hasattr(self.encoder, 'reset_parameters'):
            self.encoder.reset_parameters()
        if hasattr(self.decoder, 'reset_parameters'):
            self.decoder.reset_parameters()
        if hasattr(self.pooling_model, 'reset_parameters'):
            self.pooling_model.reset_parameters()
