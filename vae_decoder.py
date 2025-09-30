import torch
import torch.nn as nn

# A simplified EGNN layer for demonstration
class EGNNDecoderLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, feature_dim), nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim), nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.ReLU(),
            nn.Linear(feature_dim, 1, bias=False)
        )

    def forward(self, h, coords, edge_index):
        row, col = edge_index
        # Message passing
        rel_coords = coords[row] - coords[col]
        dist = torch.linalg.norm(rel_coords, dim=1, keepdim=True)
        edge_features = torch.cat([h[row], h[col], dist**2], dim=1)
        messages = self.edge_mlp(edge_features)
        
        # Aggregate messages
        agg_messages = torch.zeros_like(h)
        agg_messages.scatter_add_(0, row.unsqueeze(1).expand(-1, h.size(1)), messages)
        
        # Update node features
        h_new = self.node_mlp(torch.cat([h, agg_messages], dim=1))
        
        # Update coordinates (equivariant step)
        coord_weights = self.coord_mlp(messages)
        coord_update = torch.zeros_like(coords)
        coord_update.scatter_add_(0, row.unsqueeze(1).expand(-1, 3), coord_weights * rel_coords)
        
        return h_new, coords + coord_update

class EGNNDecoder(nn.Module):
    def __init__(self, latent_dim, num_atoms, atom_feature_dim, num_layers=4):
        super().__init__()
        self.num_atoms = num_atoms
        self.latent_dim = latent_dim
        self.atom_feature_dim = atom_feature_dim
        
        # MLP to inject latent vector
        self.latent_injector = nn.Linear(latent_dim + atom_feature_dim, atom_feature_dim)
        
        self.layers = nn.ModuleList([
            EGNNDecoderLayer(atom_feature_dim) for _ in range(num_layers)
        ])
        
    def forward(self, z, atom_types):
        """
        Args:
            z (Tensor): Latent vector, shape (batch_size, latent_dim).
            atom_types (Tensor): One-hot encoded atom types, shape (batch_size * num_atoms, atom_feature_dim).
        """
        batch_size = z.size(0)
        
        # 1. Initialize "proto-molecule"
        h = atom_types
        coords = torch.zeros(batch_size * self.num_atoms, 3, device=z.device)
        
        # 2. Inject latent vector by broadcasting
        z_expanded = z.repeat_interleave(self.num_atoms, dim=0)
        h = torch.cat([h, z_expanded], dim=1)
        h = self.latent_injector(h)

        # Create a fully connected graph for each molecule in the batch
        edge_indices = []
        for i in range(batch_size):
            offset = i * self.num_atoms
            adj = torch.ones(self.num_atoms, self.num_atoms) - torch.eye(self.num_atoms)
            edge_index = adj.to_sparse().indices() + offset
            edge_indices.append(edge_index)
        edge_index = torch.cat(edge_indices, dim=1).to(z.device)

        # 3. Iteratively refine positions
        for layer in self.layers:
            h, coords = layer(h, coords, edge_index)
            
        return coords.view(batch_size, self.num_atoms, 3)
