import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter_add


class VanillaEGNNLayer(MessagePassing):
    """Vanilla E3 Equivariant Graph Neural Network layer."""
    
    def __init__(self, hidden_dim, activation='silu'):
        super().__init__(aggr=None)
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
        row, col = edge_index
        
        # 1. Compute un-aggregated messages for each edge
        edge_messages = self.message(x[row], x[col], pos[row], pos[col])
        
        # 2. Update coordinates (Equivariant Step)
        coord_weights = self.coord_mlp(edge_messages)
        rel_pos = pos[row] - pos[col]
        
        # Aggregate coordinate updates using scatter_add
        coord_update = torch.zeros_like(pos)
        weighted_rel_pos = coord_weights * rel_pos
        coord_update.scatter_add_(0, row.unsqueeze(1).expand_as(rel_pos), weighted_rel_pos)
        
        # Add a normalization factor for stability
        num_neighbors = torch.bincount(row, minlength=pos.size(0)).float().unsqueeze(1)
        pos_new = pos + coord_update / (num_neighbors + 1e-6)
        
        # 3. Aggregate messages for node features
        aggregated_messages = scatter_add(edge_messages, row, dim=0, dim_size=x.size(0))
        
        # 4. Update node features using original features and aggregated messages
        node_mlp_input = torch.cat([x, aggregated_messages], dim=1)
        node_update = self.node_mlp(node_mlp_input)
        
        # Add residual connection and layer normalization
        x_new = self.node_norm(x + node_update)
        
        return x_new, pos_new
    
    def message(self, x_i, x_j, pos_i, pos_j):
        """
        Computes messages from node j to node i for each edge.
        
        Args:
            x_i (Tensor): Features of source nodes
            x_j (Tensor): Features of target nodes
            pos_i (Tensor): Positions of source nodes
            pos_j (Tensor): Positions of target nodes
        
        Returns:
            Tensor: Edge messages
        """
        # Compute squared distance (an invariant feature)
        dist_sq = torch.sum((pos_i - pos_j) ** 2, dim=1, keepdim=True)
        
        # Create edge features
        edge_features = torch.cat([x_i, x_j, dist_sq], dim=1)
        
        # Compute messages using the edge MLP
        return self.edge_mlp(edge_features)


class VanillaEGNNEncoder(nn.Module):
    """
    Vanilla E3 Equivariant GNN Encoder.
    
    Takes molecular coordinates and atom types as input and encodes them into a latent space.
    This encoder mirrors the decoder structure but in reverse.
    """
    
    def __init__(self, latent_dim, num_atoms, atom_feature_dim, hidden_dim=128, 
                 num_layers=6, activation='silu', pooling='mean'):
        """
        Args:
            latent_dim (int): Dimension of latent space
            num_atoms (int): Number of atoms in molecule
            atom_feature_dim (int): Dimension of atom type one-hot encoding
            hidden_dim (int): Hidden dimension for EGNN layers
            num_layers (int): Number of EGNN layers
            activation (str): Activation function ('silu' or 'relu')
            pooling (str): Pooling method ('mean', 'sum', or 'max')
        """
        super().__init__()
        self.num_atoms = num_atoms
        self.latent_dim = latent_dim
        self.atom_feature_dim = atom_feature_dim
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        
        # Initial atom embedding network
        act_fn = nn.SiLU() if activation == 'silu' else nn.ReLU()
        self.atom_embedding = nn.Sequential(
            nn.Linear(atom_feature_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Stack of EGNN layers
        self.layers = nn.ModuleList([
            VanillaEGNNLayer(hidden_dim, activation=activation) 
            for _ in range(num_layers)
        ])
        
        # Output network to map from hidden features to latent distribution parameters
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim // 2),
            act_fn,
            nn.Linear(hidden_dim // 2, latent_dim * 2)  # Output both mu and log_var
        )
        
    def forward(self, pos, atom_types, edge_index, batch):
        """
        Args:
            pos (Tensor): Atomic positions, shape (batch_size * num_atoms, 3)
            atom_types (Tensor): One-hot encoded atom types, shape (batch_size * num_atoms, atom_feature_dim)
            edge_index (Tensor): Edge connectivity, shape (2, num_edges)
            batch (Tensor): Batch assignment for each node, shape (batch_size * num_atoms,)
        
        Returns:
            mu (Tensor): Mean of latent distribution, shape (batch_size, latent_dim)
            log_var (Tensor): Log variance of latent distribution, shape (batch_size, latent_dim)
        """
        # 1. Initialize node features from atom types
        h = self.atom_embedding(atom_types)
        
        # 2. Process through EGNN layers
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)
        
        # 3. Pool node features to graph-level features
        if self.pooling == 'mean':
            graph_features = global_mean_pool(h, batch)
        elif self.pooling == 'sum':
            from torch_geometric.nn import global_add_pool
            graph_features = global_add_pool(h, batch)
        elif self.pooling == 'max':
            from torch_geometric.nn import global_max_pool
            graph_features = global_max_pool(h, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # 4. Map to latent distribution parameters
        latent_params = self.output_network(graph_features)
        
        # Split into mu and log_var
        mu = latent_params[:, :self.latent_dim]
        log_var = latent_params[:, self.latent_dim:]
        
        return mu, log_var


# For backward compatibility, create an alias
EGNNEncoder = VanillaEGNNEncoder

