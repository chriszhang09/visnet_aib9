# AIB9 VAE Training Guide

## What's New: Validation & Visualization

The training script now includes automatic validation and visualization with Weights & Biases (wandb).

## Features

### 1. **3D Molecular Visualizations**
Every 5 epochs, the script generates and logs:
- **Original molecule** from dataset
- **Reconstructed molecule** (encode → decode same molecule)
- **Generated molecule** (decode random latent vector)

### 2. **Structural Metrics**
Tracks chemical validity through:
- **RMSD**: Root mean square deviation between original and reconstructed
- **Bond lengths**: Histogram comparisons showing if bonds are realistic
- **Latent space statistics**: Mean and std of learned representations

### 3. **Wandb Integration**
All metrics and visualizations logged to wandb dashboard:
- Training loss (total, reconstruction, KL divergence)
- Validation RMSD over epochs
- Bond length distributions
- 3D molecule visualizations

## Running Training

```bash
cd /Users/chriszhang/Documents/aib9_vanillavae/visnet_aib9
python train_vae.py
```

### First Time Setup

1. Install wandb (if not already installed):
```bash
pip install wandb
```

2. Login to wandb:
```bash
wandb login
```

3. Run training - it will create a project called "aib9-vae"

## What to Look For in Wandb

### 🟢 Good Signs:
- **RMSD decreases** over epochs (reconstructions getting better)
- **Bond lengths** in reconstructed/generated molecules match original distribution
  - C-C bonds: ~1.5 Å
  - C-N bonds: ~1.3-1.5 Å  
  - C-O bonds: ~1.2-1.4 Å
- **Reconstructed molecules** look structurally similar to originals
- **Generated molecules** have chemically reasonable geometries

### 🔴 Warning Signs:
- RMSD stays high or increases
- Bond lengths drift to unrealistic values (too short <0.8 Å or too long >2.5 Å)
- Generated molecules are collapsed (all atoms at origin)
- Bond length histogram shows multiple peaks (should be ~1 peak per bond type)

## Key Metrics Dashboard

In wandb, focus on these panels:

1. **Training Losses**
   - `train/total_loss` - Overall VAE loss
   - `train/reconstruction_loss` - How well it reconstructs
   - `train/kl_divergence` - Latent space regularization

2. **Validation Metrics**
   - `val/rmsd` - Lower is better (target: <0.5 Å)
   - `val/mean_bond_length_reconstructed` - Should be ~1.2-1.5 Å
   - `val/mean_bond_length_generated` - Should match original

3. **Visualizations** (Media tab)
   - `molecules/original` - Ground truth
   - `molecules/reconstructed` - Model reconstruction
   - `molecules/generated` - Sampled from latent space
   - `molecules/bond_lengths` - Histogram comparisons

## Model Configuration

Current settings:
```python
Encoder: ViSNetBlock
  - Hidden channels: 256
  - Layers: 9
  - Uses covalent bond edges (~100-200 edges)

Decoder: EGNN
  - Hidden channels: 256
  - Layers: 8
  - Uses covalent bond edges

Latent space: 30 dimensions
Batch size: 128
Learning rate: 1e-3
```

## Troubleshooting

### "Collapsed" generated molecules (all atoms at origin)
- Decoder not learning - may need to increase decoder capacity or learning rate
- Check if KL divergence is too strong (try KL annealing)

### High RMSD not decreasing
- Model underfitting - increase model size or train longer
- Check if bond edges are properly loaded

### Unrealistic bond lengths
- Decoder not respecting chemistry - verify edge_index is correct
- May need to add bond length loss term

## Saving and Loading Models

Final model saved as `model_final.pth` and uploaded to wandb artifacts.

To load:
```python
model.load_state_dict(torch.load('model_final.pth'))
```

## Next Steps After Training

Once you have a trained model with good metrics:

1. **Sample diverse conformations**: Generate many samples and analyze
2. **Latent space interpolation**: Interpolate between two molecules
3. **Conditional generation**: Condition on specific properties
4. **Free energy analysis**: Use with aib9 tools for PMF analysis

