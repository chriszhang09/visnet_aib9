import torch
import torch.nn as nn

# Enhanced EGNN layer with deeper MLPs and more capacity
class EGNNDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, activation='silu'):
        super().__init__()
        act_fn = nn.SiLU() if activation == 'silu' else nn.ReLU()
        
        # Deeper edge MLP with residual-like structure
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim * 2),
            act_fn,
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Deeper node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            act_fn,
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Coordinate update MLP
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        # Layer normalization for stability
        self.node_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, h, coords, edge_index):
        row, col = edge_index
        
        # Message passing
        rel_coords = coords[row] - coords[col]
        dist = torch.linalg.norm(rel_coords, dim=1, keepdim=True)
        
        # Edge features with squared distance
        edge_features = torch.cat([h[row], h[col], dist**2], dim=1)
        messages = self.edge_mlp(edge_features)
        
        # Aggregate messages
        agg_messages = torch.zeros_like(h)
        # Ensure dtype consistency for scatter operation
        messages = messages.to(h.dtype)
        agg_messages.scatter_add_(0, row.unsqueeze(1).expand(-1, h.size(1)), messages)
        
        # Update node features with residual connection
        h_update = self.node_mlp(torch.cat([h, agg_messages], dim=1))
        h_new = self.node_norm(h + h_update)  # Residual connection
        
        # Update coordinates (equivariant step)
        coord_weights = self.coord_mlp(messages)
        coord_update = torch.zeros_like(coords)
        coord_update.scatter_add_(0, row.unsqueeze(1).expand(-1, 3), coord_weights * rel_coords)
        
        return h_new, coords + coord_update


class EGNNDecoder(nn.Module):
    def __init__(self, latent_dim, num_atoms, atom_feature_dim, hidden_dim=128, 
                 num_layers=6, edge_index_template=None, activation='silu'):
        """
        Enhanced EGNN Decoder with more capacity.
        
        Args:
            latent_dim (int): Dimension of latent space
            num_atoms (int): Number of atoms in molecule
            atom_feature_dim (int): Dimension of atom type one-hot encoding
            hidden_dim (int): Hidden dimension for EGNN layers (default: 128)
            num_layers (int): Number of EGNN layers (default: 6)
            edge_index_template (Tensor): Fixed edge connectivity (e.g., covalent bonds)
            activation (str): Activation function ('silu' or 'relu')
        """
        super().__init__()
        self.num_atoms = num_atoms
        self.latent_dim = latent_dim
        self.atom_feature_dim = atom_feature_dim
        self.hidden_dim = hidden_dim
        
        # Store fixed edge template if provided
        if edge_index_template is not None:
            self.register_buffer('edge_index_template', edge_index_template)
        else:
            self.edge_index_template = None
        
        # Multi-layer latent injection network
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
            EGNNDecoderLayer(hidden_dim, activation=activation) 
            for _ in range(num_layers)
        ])
        
        # Final coordinate refinement (optional)
        self.coord_refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            act_fn,
            nn.Linear(hidden_dim // 2, 3)
        )
        
    def forward(self, z, atom_types):
        """
        Args:
            z (Tensor): Latent vector, shape (batch_size, latent_dim).
            atom_types (Tensor): One-hot encoded atom types, shape (batch_size * num_atoms, atom_feature_dim).
        """
        batch_size = z.size(0)
        
        # 1. Initialize node features from atom types and latent code
        z_expanded = z.repeat_interleave(self.num_atoms, dim=0)
        h = torch.cat([atom_types, z_expanded], dim=1)
        h = self.latent_injector(h)
        
        # 2. Initialize coordinates with small random noise
        # This helps break symmetry and allows the network to learn structure
        coords = torch.randn(batch_size * self.num_atoms, 3, device=z.device) * 0.1
        
        # 3. Create edge index (use fixed template if available, otherwise fully connected)
        if self.edge_index_template is not None:
            # Use covalent bond structure
            edge_indices = []
            for i in range(batch_size):
                offset = i * self.num_atoms
                edge_indices.append(self.edge_index_template + offset)
            edge_index = torch.cat(edge_indices, dim=1)
        else:
            # Fully connected graph (fallback)
            edge_indices = []
            for i in range(batch_size):
                offset = i * self.num_atoms
                adj = torch.ones(self.num_atoms, self.num_atoms) - torch.eye(self.num_atoms)
                edge_index = adj.to_sparse().indices() + offset
                edge_indices.append(edge_index)
            edge_index = torch.cat(edge_indices, dim=1).to(z.device)
        
        # 4. Iteratively refine structure through EGNN layers
        for layer in self.layers:
            h, coords = layer(h, coords, edge_index)
        
        # 5. Optional: Final coordinate refinement based on learned features
        # coord_adjustment = self.coord_refine(h)
        # coords = coords + coord_adjustment
        
        return coords.view(batch_size, self.num_atoms, 3)
