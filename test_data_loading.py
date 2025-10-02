#!/usr/bin/env python3

import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from pathlib import Path
import sys

# Add the parent directory to the path to import aib9_tools
sys.path.append(str(Path(__file__).resolve().parent.parent))
from aib9_lib import aib9_tools as aib9

# Constants
ATOM_COUNT = 58
ATOMIC_NUMBERS = [1, 1, 1, 6, 7, 1, 6, 8, 6, 1, 1, 1, 6, 7, 1, 6, 8, 6, 1, 1, 1, 6, 7, 1, 6, 8, 6, 1, 1, 1, 6, 7, 1, 6, 8, 6, 1, 1, 1, 6, 7, 1, 6, 8, 6, 1, 1, 1, 6, 7, 1, 6, 8, 6, 1, 1, 1, 6]

def test_data_loading():
    print("Testing AIB9 data loading...")
    
    # Load data
    train_data_np = np.load(aib9.FULL_DATA)
    train_data_np = train_data_np.reshape(-1, 58, 3)
    print(f"Loaded data shape: {train_data_np.shape}")
    
    # Load topology
    topo = np.load(aib9.TOPO_FILE, allow_pickle=True)
    print(f"Topology shape: {topo.shape}")
    print(f"First few atoms: {topo[:5]}")
    
    # Create edge index
    edges = aib9.identify_all_covalent_edges(topo)
    print(f"Edges shape: {np.array(edges).shape}")
    print(f"First few edges: {edges[:5]}")
    
    # Create atomic numbers tensor
    z = torch.tensor(ATOMIC_NUMBERS, dtype=torch.long)
    print(f"Atomic numbers shape: {z.shape}")
    
    # Create edge index tensor
    edge_index = torch.tensor(edges, dtype=torch.long).contiguous()
    print(f"Edge index shape: {edge_index.shape}")
    
    # Test creating a single Data object
    print("\nTesting single Data object...")
    pos = torch.from_numpy(train_data_np[0]).float()
    print(f"Position shape: {pos.shape}")
    
    try:
        data = Data(z=z, pos=pos, edge_index=edge_index)
        print(f"✓ Single Data object created successfully")
        print(f"  z: {data.z.shape}, pos: {data.pos.shape}, edge_index: {data.edge_index.shape}")
    except Exception as e:
        print(f"✗ Failed to create single Data object: {e}")
        return False
    
    # Test creating multiple Data objects
    print("\nTesting multiple Data objects...")
    data_list = []
    for i in range(min(10, train_data_np.shape[0])):
        pos = torch.from_numpy(train_data_np[i]).float()
        data = Data(z=z.clone(), pos=pos, edge_index=edge_index.clone())
        data_list.append(data)
    
    print(f"✓ Created {len(data_list)} Data objects")
    
    # Test DataLoader with small batch
    print("\nTesting DataLoader...")
    try:
        loader = DataLoader(data_list, batch_size=2, shuffle=False, num_workers=0)
        print(f"✓ DataLoader created successfully")
        
        # Test getting first batch
        batch = next(iter(loader))
        print(f"✓ First batch loaded successfully")
        print(f"  Batch z: {batch.z.shape}, pos: {batch.pos.shape}, edge_index: {batch.edge_index.shape}")
        print(f"  Batch info: {batch.batch.shape if hasattr(batch, 'batch') else 'No batch attr'}")
        
        return True
        
    except Exception as e:
        print(f"✗ DataLoader failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
