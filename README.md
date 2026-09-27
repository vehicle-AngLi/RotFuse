# RotFuse
This directory contains the minimal inference release for **“A generic lightweight multi-modal image information fusion architecture”** (Neurocomputing, 2027). It loads the released checkpoint and fuses one aligned visible/SWIR/LWIR image triplet into one RGB image. Training, comparison-baseline, segmentation, and detection code are intentionally outside the scope of this minimal inference package.

# These sources will be released within 1 month after acceptance
- [x] ~~Network for Naive Tri-modal Fusion Baselines~~
- [x] ~~collected Dataset SEUS~~
- [x] ~~Codes (Currently under preparation and will be released before October 1, 2026)~~
- [x] ~~Checkpoint (Will be released together with the source code before October 1, 2026)~~

# Available for SEUS Dataset
You can download the dataset at:
 - https://pan.baidu.com/s/1BPa3Va1H79OeT0QDM1sD4A?pwd=euw7
 - By using the password: euw7

## Contents

- `rotfuse.py`: RotFuse network definition.
- `inference.py`: folder-based inference entry point.
- `checkpoints/rotfuse_seus.pth`: pretrained checkpoint for the SEUS dataset.
- `requirements.txt`: minimal Python dependencies.
- `LICENSE`: license retained from the original codebase on which this implementation is based.

## Installation

Python 3.8 and PyTorch 2.0.1 were used for the experiments reported in the paper. A minimal environment can be installed with:

```bash
pip install -r requirements.txt
```

## Data layout

The three input directories must contain spatially aligned images. Files are paired by filename stem, so extensions may differ:

```text
data/
├── visible/
│   ├── 0001.jpg
│   └── 0002.jpg
├── swir/
│   ├── 0001.png
│   └── 0002.png
└── lwir/
    ├── 0001.png
    └── 0002.png
```

Visible images are read as RGB. SWIR and LWIR images are read as single-channel grayscale. All three images in a triplet must have the same height and width; SEUS images in the paper are registered at 640 x 432 pixels.

## Inference

Run from this directory:

```bash
python inference.py \
  --visible data/visible \
  --swir data/swir \
  --lwir data/lwir \
  --output results
```

The script automatically uses CUDA when available. To force CPU inference, add `--device cpu`. Output images retain the chrominance channels of the visible image and use the fused luminance predicted by RotFuse.

## Citation

```bibtex
@article{li2027generic,
  title   = {A generic lightweight multi-modal image information fusion architecture},
  author  = {Li, Ang and Wu, Suzheng and Wang, Weihua and Wang, Fanxun and Liang, Jinhao and Yin, Guodong},
  journal = {Neurocomputing},
  volume  = {707},
  pages   = {135020},
  year    = {2027},
  doi     = {10.1016/j.neucom.2026.135020}
}
```
