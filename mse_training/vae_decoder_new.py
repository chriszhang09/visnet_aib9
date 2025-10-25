import torch
import torch.nn as nn
from torch_scatter import scatter
from torch_geometric.nn import knn_graph
from torch_geometric.nn import radius_graph
# --- Helper MLP ---
# A simple sequential MLP
def build_mlp(dim_in, dim_hidden, dim_out, num_layers=2, act=nn.SiLU()):
    layers = [nn.Linear(dim_in, dim_hidden), act]
    for _ in range(num_layers - 2):
        layers.extend([nn.Linear(dim_hidden, dim_hidden), act])
    layers.append(nn.Linear(dim_hidden, dim_out))
    return nn.Sequential(*layers)

class EquivariantDecoderBlock(nn.Module):
    """
    An E(3) Equivariant Message Passing Block.
    
    This block updates scalar features (s), vector features (v), and coordinates (pos)
    in an equivariant manner.
    """
    def __init__(self, hidden_dim, pos_update_weight=1e-3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_update_weight = pos_update_weight # For stable training

        # EGNN-style edge network:
        # Input: s_i, s_j, ||pos_i - pos_j||^2
        edge_input_dim = 2 * hidden_dim + 1
        self.edge_mlp = build_mlp(edge_input_dim, hidden_dim, hidden_dim)

        # EGNN-style position update network:
        # Input: phi_e (from edge_mlp)
        self.pos_update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Projector for the vector message to be used in position update
        self.vector_pos_proj = nn.Linear(hidden_dim, 1, bias=False)

        # PaiNN-style vector message network:
        # We create a scalar weight from s_j to scale v_j
        self.vector_message_mlp = build_mlp(hidden_dim, hidden_dim, hidden_dim)

        # PaiNN-style scalar update network:
        # Input: s_i, m_s (scalar message)
        self.scalar_update_mlp = build_mlp(2 * hidden_dim, hidden_dim, hidden_dim)

        # PaiNN-style vector update network:
        # We use a gate from the new scalar features to update the vector features
        self.vector_gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, s, v, pos, edges):
        """
        s: [N, H] (scalar features)
        v: [N, H, 3] (vector features)
        pos: [N, 3] (coordinates)
        edges: [2, E] (edge index)
        """
        N = s.size(0)
        i, j = edges[0], edges[1]

        # 1. Calculate Edge Information (all invariant)
        r_ij = pos[i] - pos[j]
        dist_sq = (r_ij**2).sum(dim=1, keepdim=True) # ||pos_i - pos_j||^2
        dist = torch.sqrt(dist_sq + 1e-8)
        dir_ij = r_ij / dist # [E, 3] - equivariant direction vector

        # 2. Scalar Message Path (EGNN-style)
        edge_input = torch.cat([s[i], s[j], dist_sq], dim=1)
        phi_e = self.edge_mlp(edge_input) # [E, H]

        m_s = scatter(phi_e, i, dim=0, dim_size=N, reduce='sum') # [N, H]
        
        # 3. Vector Message Path (PaiNN-style)
        # Create equivariant vector messages: scalar_weight * vector
        # This is one of many ways to do this.
        # Here, we scale the neighbor's vector feature v[j] by an MLP of its scalar feature s[j]
        vector_weights = self.vector_message_mlp(s[j]) # [E, H]
        
        # We scale v[j] by the computed weights. This is equivariant.
        # [E, H, 3] * [E, H, 1] -> [E, H, 3]
        v_message = v[j] * vector_weights.unsqueeze(-1)
        
        m_v = scatter(v_message, i, dim=0, dim_size=N, reduce='sum') # [N, H, 3]

        # 4. Position Update (EGNN + Vector Features)
        # Compute an invariant scalar weight for the direction vector (from scalar features)
        pos_weights_s = self.pos_update_mlp(phi_e) # [E, 1]
        pos_update_msg_s = pos_weights_s * dir_ij # [E, 3]
        
        # Create an equivariant position update from the vector features
        # Project [E, H, 3] -> [E, 3, H]
        v_msg_for_pos = v_message.transpose(1, 2)
        # Project [E, 3, H] -> [E, 3, 1] -> [E, 3]
        pos_update_msg_v = self.vector_pos_proj(v_msg_for_pos).squeeze(-1)

        # Combine the two update messages
        pos_update_msg = pos_update_msg_s + pos_update_msg_v
        
        # Aggregate (sum) messages and update positions.
        # We use 'mean' for stability, preventing explosion.
        pos_delta = scatter(pos_update_msg, i, dim=0, dim_size=N, reduce='mean')
        pos_new = pos + pos_delta * self.pos_update_weight

        # 5. Feature Updates
        # Update scalar features
        s_new = s + self.scalar_update_mlp(torch.cat([s, m_s], dim=1)) # [N, H]

        # Update vector features (gated)
        v_gate = torch.sigmoid(self.vector_gate(s_new)) # [N, H]
        v_new = v + m_v * v_gate.unsqueeze(-1) # [N, H, 3]

        return s_new, v_new, pos_new


class EquivariantDecoder(nn.Module):
    """
    E(3) Equivariant VAE Decoder.
    
    Takes latent scalars (z_s), latent vectors (z_v), and random positions (pos_rand)
    and reconstructs the 3D geometry.
    """
    def __init__(self, scalar_lat_dim, vector_lat_dim, hidden_dim, num_layers=4, k_neighbors=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors

        # 1. Project latent codes to hidden dimension
        self.scalar_proj = nn.Linear(scalar_lat_dim, hidden_dim)
        
        # This is an equivariant "linear" layer.
        # It projects the feature dimension (F_v -> H) while leaving the 3 spatial dims alone.
        self.vector_proj = nn.Linear(vector_lat_dim, hidden_dim)

        # 2. Stack of equivariant message passing blocks
        self.layers = nn.ModuleList(
            [EquivariantDecoderBlock(hidden_dim) for _ in range(num_layers)]
        )

    def forward(self, z_s, z_v, pos_rand):
        """
        z_s: [N, F_s] (Scalar latent code)
        z_v: [N, F_v, 3] (Vector latent code)
        pos_rand: [N, 3] (Random initial positions)
        """
        
        # 1. Initial Embedding
        s = self.scalar_proj(z_s) # [N, H]
        
        # Project vector features:
        # (N, F_v, 3) -> (N, 3, F_v)
        v_trans = z_v.transpose(1, 2)
        # (N, 3, F_v) -> (N, 3, H)
        v_proj = self.vector_proj(v_trans)
        # (N, 3, H) -> (N, H, 3)
        v = v_proj.transpose(1, 2)
        
        pos = pos_rand

        # 2. Run Message Passing Layers
        for layer in self.layers:
            # We re-compute the graph at each step based on the updated positions
            # This makes the graph dynamic
            # Move to CPU for knn_graph, then back to original device
            device = pos.device
            edges = radius_graph(pos, r=5.0, batch=torch.zeros(pos.shape[0], dtype=torch.long).to(device))
            
            # Update all features and positions
            s, v, pos = layer(s, v, pos, edges)

        # 3. Return the final reconstructed positions
        return pos

# =============================================================================
# --- EQUIVARIANCE TEST ---
# =============================================================================
if __name__ == "__main__":
    print("--- Running E(3) Equivariance Test for Decoder ---")

    # 1. Setup Model and Inputs
    N = 10         # Number of nodes
    F_s = 310       # Scalar latent features
    F_v = 16       # Vector latent features
    H = 128        # Hidden dimension
    N_layers = 3
    K = N - 1      # Use a fully-connected graph (k=N-1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize the decoder
    decoder = EquivariantDecoder(
        scalar_lat_dim=F_s,
        vector_lat_dim=F_v,
        hidden_dim=H,
        num_layers=N_layers,
        k_neighbors=K
    ).to(device).double() # <-- Convert model parameters to double
    decoder.eval()

    # Create random inputs
    pos_rand = torch.randn(N, 3).to(device).double() # <-- Use double
    z_s = torch.randn(N, F_s).to(device).double() # <-- Use double
    z_v = torch.randn(N, F_v, 3).to(device).double() # <-- Use double

    # 2. Generate a Random 3D Rotation Matrix (R)
    A = torch.randn(3, 3).to(device).double() # <-- Use double
    R, _ = torch.linalg.qr(A)
    if torch.det(R) < 0:
        R[:, 0] *= -1  # Ensure determinant is +1
    
    print("\nGenerated random 3x3 rotation matrix R.")

    # 3. Path 1: (Decode -> Rotate)
    with torch.no_grad():
        X_out_1 = decoder(z_s, z_v, pos_rand)
    
    # Rotate the final output coordinates
    X_out_1_rotated = torch.matmul(X_out_1, R)
    print("Path 1 (Decode -> Rotate) complete.")

    # 4. Path 2: (Rotate -> Decode)
    # Rotate the equivariant inputs
    pos_rand_rotated = torch.matmul(pos_rand, R)
    z_v_rotated = torch.matmul(z_v, R)
    # z_s is invariant, so it is NOT rotated

    with torch.no_grad():
        X_out_2 = decoder(z_s, z_v_rotated, pos_rand_rotated)
    print("Path 2 (Rotate -> Decode) complete.")

    # 5. Compare the Results
    tolerance = 1e-5
    is_equivariant = torch.allclose(
        X_out_1_rotated, 
        X_out_2, 
        atol=tolerance
    )

    print("\n--- TEST RESULTS ---")
    print(f"Decoder is equivariant: {is_equivariant}")

    if is_equivariant:
        print("Test PASSED!")
    else:
        print("Test FAILED!")
        
    print(f"Max difference between outputs: {(X_out_1_rotated - X_out_2).abs().max().item()}")
    print("Note: A very small difference (e.g., < 1e-5) is expected due to floating point error.")


