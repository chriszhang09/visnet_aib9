import torch
import torch.nn as nn
import torch.nn.functional as F
from .vanilla_encoder import VanillaEGNNEncoder


class VanillaEGNNLayer(nn.Module):
    """Vanilla E3 Equivariant Graph Neural Network layer (for decoder)."""
    
    def __init__(self, hidden_dim, activation='silu'):
        super().__init__()
        act_fn = nn.SiLU() if activation == 'silu' else nn.ReLU()
        
        # Edge MLP for computing messages from edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),  # Input: [h_i, h_j, ||x_i - x_j||^2]
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn
        )
        
        # Node MLP for updating node features from aggregated messages
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Input: [h_i, m_i]
            act_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Coordinate MLP for computing coordinate weights from edge messages
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1, bias=False)  # Output a single scalar weight
        )
        
        # Layer normalization for stability
        self.node_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, pos, edge_index):
        """
        Args:
            x (Tensor): Node features [num_nodes, hidden_dim]
            pos (Tensor): Node positions [num_nodes, 3]
            edge_index (LongTensor): Edge connectivity [2, num_edges]
        
        Returns:
            x_new (Tensor): Updated node features [num_nodes, hidden_dim]
            pos_new (Tensor): Updated positions [num_nodes, 3]
        """
        from torch_scatter import scatter_add
        
        row, col = edge_index
        
        # 1. Compute edge messages
        edge_messages = self.message(x[row], x[col], pos[row], pos[col])
        
        # 2. Update coordinates (Equivariant Step)
        coord_weights = self.coord_mlp(edge_messages)
        rel_pos = pos[row] - pos[col]
        
        # Aggregate coordinate updates
        coord_update = torch.zeros_like(pos)
        weighted_rel_pos = coord_weights * rel_pos
        coord_update.scatter_add_(0, row.unsqueeze(1).expand_as(rel_pos), weighted_rel_pos)
        
        # Normalize by number of neighbors
        num_neighbors = torch.bincount(row, minlength=pos.size(0)).float().unsqueeze(1)
        pos_new = pos + coord_update / (num_neighbors + 1e-6)
        
        # 3. Aggregate messages for node features
        aggregated_messages = scatter_add(edge_messages, row, dim=0, dim_size=x.size(0))
        
        # 4. Update node features
        node_mlp_input = torch.cat([x, aggregated_messages], dim=1)
        node_update = self.node_mlp(node_mlp_input)
        
        # Add residual connection and layer normalization
        x_new = self.node_norm(x + node_update)
        
        return x_new, pos_new
    
    def message(self, x_i, x_j, pos_i, pos_j):
        """Compute messages from node j to node i for each edge."""
        # Compute squared distance (an invariant feature)
        dist_sq = torch.sum((pos_i - pos_j) ** 2, dim=1, keepdim=True)
        
        # Create edge features
        edge_features = torch.cat([x_i, x_j, dist_sq], dim=1)
        
        # Compute messages using the edge MLP
        return self.edge_mlp(edge_features)


class VanillaEGNNDecoder(nn.Module):
    """
    Vanilla E3 Equivariant GNN Decoder.
    
    Takes latent codes and generates molecular coordinates.
    """
    
    def __init__(self, latent_dim, num_atoms, atom_feature_dim, hidden_dim=128, 
                 num_layers=6, activation='silu'):
        """
        Args:
            latent_dim (int): Dimension of latent space
            num_atoms (int): Number of atoms in molecule
            atom_feature_dim (int): Dimension of atom type one-hot encoding
            hidden_dim (int): Hidden dimension for EGNN layers
            num_layers (int): Number of EGNN layers
            activation (str): Activation function ('silu' or 'relu')
        """
        super().__init__()
        self.num_atoms = num_atoms
        self.latent_dim = latent_dim
        self.atom_feature_dim = atom_feature_dim
        self.hidden_dim = hidden_dim
        
        # Latent injection network
        act_fn = nn.SiLU() if activation == 'silu' else nn.ReLU()
        self.latent_injector = nn.Sequential(
            nn.Linear(latent_dim + atom_feature_dim, hidden_dim * 2),
            act_fn,
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Stack of EGNN layers
        self.layers = nn.ModuleList([
            VanillaEGNNLayer(hidden_dim, activation=activation) 
            for _ in range(num_layers)
        ])
        
    def forward(self, z, atom_types, edge_index, batch):
        """
        Args:
            z (Tensor): Latent vectors, shape (batch_size, latent_dim)
            atom_types (Tensor): One-hot encoded atom types, shape (batch_size * num_atoms, atom_feature_dim)
            edge_index (Tensor): Edge connectivity, shape (2, num_edges)
            batch (Tensor): Batch assignment for each node, shape (batch_size * num_atoms,)
        
        Returns:
            Tensor: Reconstructed positions, shape (batch_size * num_atoms, 3)
        """
        batch_size = z.size(0)
        
        # 1. Expand latent codes to all atoms in the batch
        z_expanded = z.repeat_interleave(self.num_atoms, dim=0)
        
        # 2. Initialize node features by combining atom types and latent codes
        h = torch.cat([atom_types, z_expanded], dim=1)
        h = self.latent_injector(h)
        
        # 3. Initialize coordinates with small random noise
        pos = torch.randn(batch_size * self.num_atoms, 3, device=z.device) * 0.1
        
        # 4. Iteratively refine structure through EGNN layers
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)
        
        return pos


class MolecularVAEVanilla(nn.Module):
    """
    Vanilla Molecular VAE using E3 equivariant GNN for both encoder and decoder.
    
    This model uses simple, vanilla EGNN architecture for both encoding and decoding,
    making it a fully symmetric and interpretable VAE.
    """
    
    def __init__(self, latent_dim, num_atoms, atom_feature_dim, 
                 encoder_hidden_dim=128, decoder_hidden_dim=128, 
                 encoder_num_layers=6, decoder_num_layers=6, 
                 encoder_pooling='mean', activation='silu'):
        """
        Args:
            latent_dim (int): Dimension of the latent space
            num_atoms (int): Number of atoms in each molecule
            atom_feature_dim (int): Dimension for one-hot encoding of atom types (max_atomic_number + 1)
            encoder_hidden_dim (int): Hidden dimension for EGNN encoder
            decoder_hidden_dim (int): Hidden dimension for EGNN decoder
            encoder_num_layers (int): Number of EGNN layers in encoder
            decoder_num_layers (int): Number of EGNN layers in decoder
            encoder_pooling (str): Pooling method for encoder ('mean', 'sum', or 'max')
            activation (str): Activation function ('silu' or 'relu')
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_atoms = num_atoms
        self.atom_feature_dim = atom_feature_dim
        
        # Vanilla EGNN Encoder
        self.encoder = VanillaEGNNEncoder(
            latent_dim=latent_dim,
            num_atoms=num_atoms,
            atom_feature_dim=atom_feature_dim,
            hidden_dim=encoder_hidden_dim,
            num_layers=encoder_num_layers,
            activation=activation,
            pooling=encoder_pooling
        )
        
        # Vanilla EGNN Decoder
        self.decoder = VanillaEGNNDecoder(
            latent_dim=latent_dim,
            num_atoms=num_atoms,
            atom_feature_dim=atom_feature_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            activation=activation
        )

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, I)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def sample_prior(self, batch_size, device):
        """Sample from standard Gaussian prior N(0, I)"""
        return torch.randn(batch_size, self.latent_dim, device=device)

    def forward(self, data):
        """
        Forward pass through the VAE.
        
        Args:
            data: PyTorch Geometric Data object with:
                - pos: Atomic positions [batch_size * num_atoms, 3]
                - z: Atom types [batch_size * num_atoms]
                - edge_index: Edge connectivity [2, num_edges]
                - batch: Batch indices [batch_size * num_atoms]
        
        Returns:
            reconstructed_pos: Reconstructed positions [batch_size * num_atoms, 3]
            mu: Latent mean [batch_size, latent_dim]
            log_var: Latent log variance [batch_size, latent_dim]
        """
        # Convert atom types to one-hot encoding
        atom_types_one_hot = F.one_hot(data.z.long(), num_classes=self.atom_feature_dim).float()
        
        # Encode
        mu, log_var = self.encoder(data.pos, atom_types_one_hot, data.edge_index, data.batch)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        reconstructed_pos = self.decoder(z, atom_types_one_hot, data.edge_index, data.batch)
        
        return reconstructed_pos, mu, log_var


def pairwise_distance_loss_vanilla(pred_coords, target_coords, batch=None):
    """
    E(3) invariant loss using all pairwise distances between atoms.
    
    Args:
        pred_coords: Predicted coordinates [batch_size * num_atoms, 3]
        target_coords: Target coordinates [batch_size * num_atoms, 3]
        batch: Batch indices [batch_size * num_atoms] (optional)
    
    Returns:
        Scalar loss value
    """
    # Convert to float32 for numerical stability
    pred_coords = pred_coords.float()
    target_coords = target_coords.float()
    
    if batch is not None:
        # Handle batched data
        batch_size = batch.max().item() + 1
        total_loss = 0
        
        for i in range(batch_size):
            # Get coordinates for this molecule
            mask = (batch == i)
            pred_mol = pred_coords[mask]
            target_mol = target_coords[mask]
            
            # Compute pairwise distances
            pred_dists = torch.cdist(pred_mol, pred_mol)
            target_dists = torch.cdist(target_mol, target_mol)
            
            # Only use upper triangular part (avoid diagonal and duplicates)
            triu_mask = torch.triu(torch.ones_like(pred_dists), diagonal=1).bool()
            pred_dists_upper = pred_dists[triu_mask]
            target_dists_upper = target_dists[triu_mask]
            
            # MSE on pairwise distances
            mol_loss = F.mse_loss(pred_dists_upper, target_dists_upper)
            total_loss += mol_loss
        
        return total_loss / batch_size
    else:
        # Single molecule
        pred_dists = torch.cdist(pred_coords, pred_coords)
        target_dists = torch.cdist(target_coords, target_coords)
        
        # Only use upper triangular part
        triu_mask = torch.triu(torch.ones_like(pred_dists), diagonal=1).bool()
        pred_dists_upper = pred_dists[triu_mask]
        target_dists_upper = target_dists[triu_mask]
        
        return F.mse_loss(pred_dists_upper, target_dists_upper)


def vae_loss_function_vanilla(reconstructed_pos, original_pos, mu, log_var, 
                               batch=None, beta=1.0):
    """
    Complete VAE loss function for vanilla model.
    
    Args:
        reconstructed_pos: Predicted coordinates [batch_size * num_atoms, 3]
        original_pos: Ground truth coordinates [batch_size * num_atoms, 3]
        mu: Latent mean [batch_size, latent_dim]
        log_var: Latent log variance [batch_size, latent_dim]
        batch: Batch indices [batch_size * num_atoms] (optional)
        beta: Weight for KL divergence term (beta-VAE)
    
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss component
        kl_loss: KL divergence loss component
    """
    # Reconstruction loss using pairwise distances (E(3) invariant)
    recon_loss = pairwise_distance_loss_vanilla(reconstructed_pos, original_pos, batch)
    
    # KL divergence: KL(N(μ, σ²) || N(0, I))
    # = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_loss = kl_loss / mu.size(0)  # Average over batch
    
    # Total loss with beta weighting
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss

