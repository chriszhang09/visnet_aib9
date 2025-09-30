# Bug Fix: edge_index Shape Error

## Error
```
ValueError: Expected 'edge_index' to have size '2' in the first dimension (got '112')
```

## Root Cause
In `train_vae.py`, we were incorrectly transposing the edge_index:

```python
# WRONG:
edges = aib9.identify_all_covalent_edges(topo)
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # ❌ Extra transpose!
```

The function `identify_all_covalent_edges()` already returns edges in shape `[2, num_edges]`:
- Line 508: `edge_index = np.array(edge_list, dtype=int).T`
- Returns: `[2, 112]` for 56 bidirectional bonds

When we called `.t()` again, it became `[112, 2]` which is wrong for PyTorch Geometric.

## Fix
Remove the extra transpose:

```python
# CORRECT:
edges = aib9.identify_all_covalent_edges(topo)
edge_index = torch.tensor(edges, dtype=torch.long).contiguous()  # ✓ No transpose needed!
```

## Expected Shapes
- `edge_index`: `[2, num_edges]` where `num_edges = 112` (56 bonds × 2 directions)
- First row: source nodes
- Second row: destination nodes

## Verification
Your training should now work correctly. The edge_index will be properly batched by PyTorch Geometric's DataLoader.

## File Modified
- `visnet_aib9/train_vae.py` line 236-237

