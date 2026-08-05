# Network Architectures

This document provides a detailed layer-by-layer breakdown of the three multi-modal fusion networks used in this project: **NC-Fuse**, **SED-Fuse**, and **Tri-UNet**.

---

## Table A: Naive Concat Fusion (NC-Fuse) Network
The NC-Fuse model follows an early-fusion strategy where all modalities are concatenated at the input level.

| Layer Name | Input Size $(C, W, H)$ | Key Operations | Output Size $(C, W, H)$ |
| :--- | :--- | :--- | :--- |
| **Concat** | (3,W,H); (1,W,H); (1,W,H) | Channel-wise Concatenation | (5, W, H) |
| **Head** | (5, W, H) | Conv(3×3); BN; LeakyReLU | (64, W, H) |
| **Encode1** | (64, W, H) | Conv(3×3, Stride=2); BN; LeakyReLU | (64, W/2, H/2) |
| **ResNet1** | (64, W/2, H/2) | Residual Block (2 × Conv 3×3) | (64, W/2, H/2) |
| **Encode2** | (64, W/2, H/2) | Conv(3×3, Stride=2); BN; LeakyReLU | (128, W/4, H/4) |
| **ResNet2** | (128, W/4, H/4) | Residual Block (2 × Conv 3×3) | (128, W/4, H/4) |
| **Upsample1** | (128, W/4, H/4) | 2× Bilinear Interpolation | (128, W/2, H/2) |
| **Decode1** | (128, W/2, H/2) | Conv(3×3); BN; LeakyReLU | (64, W/2, H/2) |
| **Upsample2** | (64, W/2, H/2) | 2× Bilinear Interpolation | (64, W, H) |
| **Decode2** | (64, W, H) | Conv(3×3); BN; LeakyReLU | (32, W, H) |
| **Output** | (32, W, H) | Conv(3×3); Tanh | (3, W, H) |

---

## Table B: Simple Encoder-Decoder Fusion (SED-Fuse) Network
SED-Fuse uses independent encoders for each modality and fuses them at the bottleneck (late-fusion).

| Layer Name | Input Size $(C, W, H)$ | Key Operations | Output Size $(C, W, H)$ |
| :--- | :--- | :--- | :--- |
| **Pre-Conv (L/S)** | 2 × (1, W, H) | Conv(3×3) | 2 × (3, W, H) |
| **Encoder – Block1** | (3, W, H) | Conv(3×3); BN; LeakyReLU | (32, W, H) |
| **Encoder – Block2** | (32, W, H) | Conv(3×3, Stride=2); BN; LeakyReLU | (64, W/2, H/2) |
| **Encoder – Block3** | (64, W/2, H/2) | Residual Block (2 × Conv 3×3) | (64, W/2, H/2) |
| **Encoder – Block4** | (64, W/2, H/2) | Conv(3×3, Stride=2); BN; LeakyReLU | (128, W/4, H/4) |
| **Fusion Layer** | 3 × (128, W/4, H/4) | Concat; Conv(1×1); ResBlock | (128, W/4, H/4) |
| **Decoder – Block1** | (128, W/4, H/4) | 2× Bilinear Upsample; Conv(3×3) | (64, W/2, H/2) |
| **Decoder – Block2** | (64, W/2, H/2) | 2× Bilinear Upsample; Conv(3×3) | (32, W, H) |
| **Output** | (32, W, H) | Conv(3×3); Tanh | (3, W, H) |

---

## Table C: Triple-UNet Fusion (Tri-UNet) Network
Tri-UNet implements multi-scale fusion with skip connections to preserve spatial details across all modalities.

| Layer Name | Input Size $(C, W, H)$ | Key Operations | Output Size $(C, W, H)$ |
| :--- | :--- | :--- | :--- |
| **Scale1-Encoder1** | (3,W,H); 2 × (1,W,H) | 3 × DoubleConv Branches | 3 × (32, W, H) |
| **Scale1-Fuse (skip1)** | 3 × (32, W, H) | Concat [V/L/S]; Conv(1×1) | (32, W, H) |
| **Scale2-Downsample1** | 3 × (32, W, H) | 3 × MaxPool(2×2) | 3 × (32, W/2, H/2) |
| **Scale2-Encoder2** | 3 × (32, W/2, H/2) | 3 × DoubleConv Branches | 3 × (64, W/2, H/2) |
| **Scale2-Fuse (skip2)** | 3 × (64, W/2, H/2) | Concat [V/L/S]; Conv(1×1) | (64, W/2, H/2) |
| **Downsample2 (DS2)** | 3 × (64, W/2, H/2) | 3 × MaxPool(2×2) | 3 × (64, W/4, H/4) |
| **Bottleneck** | 3 × (64, W/4, H/4) | Concat [V/L/S]; DoubleConv | (128, W/4, H/4) |
| **Upsample1 (Up1)** | (128, W/4, H/4) | Transpose Conv (4×4, Stride=2) | (64, W/2, H/2) |
| **Decoder1 (D1)** | 2 × (64, W/2, H/2) | Concat [Up1, skip2]; DoubleConv | (64, W/2, H/2) |
| **Upsample2 (Up2)** | (64, W/2, H/2) | Transpose Conv (4×4, Stride=2) | (32, W, H) |
| **Decoder2 (D2)** | 2 × (32, W, H) | Concat [Up2, skip1]; DoubleConv | (32, W, H) |
| **Output** | (32, W, H) | Conv(1×1); Tanh | (3, W, H) |

---
**Notes:**
- **DoubleConv**: Consists of `[Conv3x3 -> BatchNorm -> LeakyReLU]` repeated twice.
- **Residual Block**: A skip-connection block containing two `Conv3x3` layers.
- **Dimensions**: All sizes are represented as `(Channels, Width, Height)`.