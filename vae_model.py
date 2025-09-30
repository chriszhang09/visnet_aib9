import torch
import torch.nn as nn
import torch.nn.functional as F
from visnet_vae_encoder import ViSNetEncoder
from vae_decoder import EGNNDecoder

class MolecularVAE(nn.Module):
    def __init__(self, latent_dim, num_atoms, atom_feature_dim, 
                 visnet_hidden_channels=128, decoder_hidden_dim=128, 
                 decoder_num_layers=6, edge_index_template=None, 
                 visnet_kwargs=None):
        """
        Args:
            latent_dim (int): Dimension of the latent space
            num_atoms (int): Number of atoms in each molecule
            atom_feature_dim (int): Dimension for one-hot encoding of atom types (max_atomic_number + 1)
            visnet_hidden_channels (int): Hidden dimension for ViSNet encoder
            decoder_hidden_dim (int): Hidden dimension for EGNN decoder
            decoder_num_layers (int): Number of EGNN layers in decoder
            edge_index_template (Tensor): Fixed edge connectivity for decoder
            visnet_kwargs (dict): Additional parameters for ViSNetBlock
        """
        super().__init__()
        if visnet_kwargs is None:
            visnet_kwargs = {}
        
        self.encoder = ViSNetEncoder(
            latent_dim=latent_dim,
            visnet_hidden_channels=visnet_hidden_channels,
            **visnet_kwargs
        )
        self.decoder = EGNNDecoder(
            latent_dim=latent_dim, 
            num_atoms=num_atoms, 
            atom_feature_dim=atom_feature_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            edge_index_template=edge_index_template
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        # Encode
        mu, log_var = self.encoder(data)
        
        z = self.reparameterize(mu, log_var)
        
        # Decode
        # We need the one-hot atom types for the decoder
        # This assumes data.z contains integer atom types
        atom_types_one_hot = F.one_hot(data.z.long(), num_classes=self.decoder.atom_feature_dim).float()
        reconstructed_pos = self.decoder(z, atom_types_one_hot)
        
        return reconstructed_pos, mu, log_var

def vae_loss_function(reconstructed_pos, original_pos, mu, log_var):

    recon_loss = F.mse_loss(reconstructed_pos.view(-1, 3), original_pos)
    
    # 2. KL Divergence
    # Measures how much the learned distribution N(mu, sigma) deviates from N(0, 1)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kl_div
