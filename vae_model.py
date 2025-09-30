import torch
import torch.nn as nn
import torch.nn.functional as F
from visnet_vae_encoder import ViSNetEncoder
from vae_decoder import EGNNDecoder

# Assume ViSNetEncoder and EGNNNablaDecoder are defined as above

class MolecularVAE(nn.Module):
    def __init__(self, visnet_model, latent_dim, num_atoms, atom_feature_dim):
        super().__init__()
        self.encoder = ViSNetEncoder(visnet_model, latent_dim)
        self.decoder = EGNNDecoder(latent_dim, num_atoms, atom_feature_dim)

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
