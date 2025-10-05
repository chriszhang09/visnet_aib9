# Training Organization

This document explains the organization of the two different training approaches in this project.

## 📁 Folder Structure

```
visnet_aib9/
├── train_vae.py                    # Bond-based E(3) invariant training
├── train_vae_mse.py               # Pairwise distance MSE training
├── visnet_vae_encoder.py          # Original encoder (bond-based)
├── vae_model.py                   # Original VAE model (bond-based)
├── vae_decoder.py                 # Original decoder (bond-based)
├── vae_utils.py                   # Original utilities (bond-based)
└── mse_training/                  # MSE-specific components
    ├── __init__.py
    ├── visnet_vae_encoder_mse.py  # MSE encoder with proper initialization
    ├── vae_decoder_mse.py         # MSE decoder
    ├── vae_model_mse.py           # MSE VAE model
    └── vae_utils_mse.py          # MSE utilities
```

## 🎯 Two Training Approaches

### 1. **Bond-Based Training** (`train_vae.py`)
- **Loss Function**: E(3) invariant bond distance loss
- **Components**: Original files in root directory
- **Use Case**: When you want to focus on covalent bond accuracy
- **Encoder**: `ViSNetEncoder` (original)
- **Decoder**: `EGNNDecoder` (original)
- **Model**: `MolecularVAE` (original)

### 2. **Pairwise Distance Training** (`train_vae_mse.py`)
- **Loss Function**: E(3) invariant pairwise distance loss (all atom pairs)
- **Components**: MSE-specific files in `mse_training/` folder
- **Use Case**: When you want complete 3D structure accuracy
- **Encoder**: `ViSNetEncoderMSE` (with proper initialization)
- **Decoder**: `EGNNDecoderMSE` (MSE-specific)
- **Model**: `MolecularVAEMSE` (MSE-specific)

## 🔧 Key Differences

### **Encoder Initialization**
- **Original**: Standard initialization
- **MSE**: Proper initialization to prevent variance explosion
  ```python
  # MSE encoder initializes log_var to reasonable values
  final_block.scalar_linear.bias.data[-1] = -1.0  # Start with reasonable variance
  final_block.scalar_linear.weight.data[-1] *= 0.1  # Scale down weights
  ```

### **Loss Functions**
- **Bond-based**: Only uses covalent bond distances (57 bonds for 58 atoms)
- **Pairwise**: Uses ALL pairwise distances (1,653 pairs for 58 atoms)

### **Training Stability**
- **Bond-based**: Generally stable, focuses on chemical connectivity
- **MSE**: More informative but requires careful initialization to prevent variance explosion

## 🚀 Usage

### Run Bond-Based Training:
```bash
python train_vae.py
```

### Run Pairwise Distance Training:
```bash
python train_vae_mse.py
```

## 📊 Expected Results

### **Bond-Based Training**:
- ✅ Stable training
- ✅ Good chemical connectivity
- ⚠️ May miss long-range interactions

### **Pairwise Distance Training**:
- ✅ Complete 3D structure accuracy
- ✅ Captures all molecular interactions
- ⚠️ Requires proper initialization
- ⚠️ More computationally expensive

## 🔄 Independence

Both training approaches are completely independent:
- ✅ No shared components
- ✅ Can run simultaneously
- ✅ Different model checkpoints
- ✅ Different W&B projects

## 🎯 Recommendations

- **Use Bond-Based** for: Quick prototyping, chemical connectivity focus
- **Use Pairwise Distance** for: Complete molecular structure, research applications

Both approaches are E(3) invariant and suitable for molecular generation tasks!

