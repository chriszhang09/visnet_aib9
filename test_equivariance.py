"""
Test E(3) equivariance of the EGNN decoder layers.
"""
import torch
import numpy as np
from vae_decoder import EGNNDecoderLayer

def random_rotation_matrix():
    """Generate a random 3D rotation matrix."""
    # Using QR decomposition of a random matrix
    H = torch.randn(3, 3)
    Q, R = torch.linalg.qr(H)
    return Q

def test_egnn_layer_equivariance():
    """Test if EGNNDecoderLayer is rotation equivariant."""
    
    # Setup
    hidden_dim = 64
    num_atoms = 10
    layer = EGNNDecoderLayer(hidden_dim, activation='silu')
    layer.eval()  # Set to eval mode
    
    # Create dummy data
    h = torch.randn(num_atoms, hidden_dim)
    coords = torch.randn(num_atoms, 3)
    
    # Create a simple edge index (chain)
    edges = [[i, i+1] for i in range(num_atoms-1)]
    edges += [[i+1, i] for i in range(num_atoms-1)]  # bidirectional
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    # Generate random rotation and translation
    R = random_rotation_matrix()
    t = torch.randn(1, 3)
    
    # Apply transformation to coordinates
    coords_transformed = (R @ coords.T).T + t
    
    # Forward pass on original
    with torch.no_grad():
        h_out1, coords_out1 = layer(h, coords, edge_index)
    
    # Forward pass on transformed
    with torch.no_grad():
        h_out2, coords_out2 = layer(h, coords_transformed, edge_index)
    
    # Check equivariance: T(f(x)) should equal f(T(x))
    coords_out1_transformed = (R @ coords_out1.T).T + t
    
    # Compute difference
    coord_diff = torch.abs(coords_out2 - coords_out1_transformed).max().item()
    feature_diff = torch.abs(h_out1 - h_out2).max().item()
    
    print("="*60)
    print("E(3) Equivariance Test for EGNNDecoderLayer")
    print("="*60)
    print(f"Max coordinate difference: {coord_diff:.6f}")
    print(f"Max feature difference: {feature_diff:.6f}")
    print()
    
    # Check invariance of features
    print("✓ Features should be INVARIANT (same before/after rotation)")
    if feature_diff < 1e-5:
        print(f"  PASS: Features are invariant (diff={feature_diff:.2e})")
    else:
        print(f"  FAIL: Features changed (diff={feature_diff:.2e})")
    print()
    
    # Check equivariance of coordinates
    print("✓ Coordinates should be EQUIVARIANT (rotate together)")
    if coord_diff < 1e-5:
        print(f"  PASS: Coordinates are equivariant (diff={coord_diff:.2e})")
    else:
        print(f"  FAIL: Coordinates not equivariant (diff={coord_diff:.2e})")
    print()
    
    # Check translation equivariance separately (without rotation)
    coords_only_translated = coords + t
    with torch.no_grad():
        h_out3, coords_out3 = layer(h, coords_only_translated, edge_index)
    
    # f(x + t) should equal f(x) + t
    translation_diff = torch.abs(coords_out3 - (coords_out1 + t)).max().item()
    
    print("✓ Translation equivariance: f(x + t) = f(x) + t")
    if translation_diff < 1e-5:
        print(f"  PASS: Translation equivariant (diff={translation_diff:.2e})")
    else:
        print(f"  FAIL: Not translation equivariant (diff={translation_diff:.2e})")
    
    print("="*60)
    
    return coord_diff < 1e-5 and feature_diff < 1e-5

def test_distance_invariance():
    """Test that distances are preserved (sanity check)."""
    print("\n" + "="*60)
    print("Distance Invariance Test")
    print("="*60)
    
    coords1 = torch.randn(10, 3)
    R = random_rotation_matrix()
    t = torch.randn(1, 3)
    coords2 = (R @ coords1.T).T + t
    
    # Compute pairwise distances
    dist1 = torch.cdist(coords1, coords1)
    dist2 = torch.cdist(coords2, coords2)
    
    diff = torch.abs(dist1 - dist2).max().item()
    print(f"Max distance difference after rotation+translation: {diff:.6e}")
    
    if diff < 1e-5:
        print("✓ PASS: Distances are preserved under E(3) transformations")
    else:
        print("✗ FAIL: Distances changed!")
    print("="*60)

if __name__ == "__main__":
    print("\n🔬 Testing E(3) Equivariance of EGNN Decoder\n")
    
    # Test basic distance invariance
    test_distance_invariance()
    
    # Test EGNN layer equivariance
    success = test_egnn_layer_equivariance()
    
    if success:
        print("\n✅ All tests passed! The EGNN layers are E(3) equivariant.")
    else:
        print("\n❌ Tests failed! Check the implementation.")

