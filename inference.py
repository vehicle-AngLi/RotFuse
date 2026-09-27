"""Run a pretrained RotFuse checkpoint on aligned tri-modal image folders."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rotfuse import RotFuse


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def image_map(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {directory}")
    files = {p.stem: p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    if not files:
        raise ValueError(f"No supported images found in: {directory}")
    return files


def load_rgb(path):
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)


def load_gray(path):
    array = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def rgb_to_ycbcr(rgb):
    red, green, blue = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    cr = (red - y) * 0.713 + 0.5
    cb = (blue - y) * 0.564 + 0.5
    return y.clamp(0, 1), cb.clamp(0, 1), cr.clamp(0, 1)


def ycbcr_to_rgb(y, cb, cr):
    red = y + 1.403 * (cr - 0.5)
    green = y - 0.714 * (cr - 0.5) - 0.344 * (cb - 0.5)
    blue = y + 1.773 * (cb - 0.5)
    return torch.cat((red, green, blue), dim=1).clamp(0, 1)


def load_checkpoint(model, checkpoint, device):
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 1.13
        state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visible", required=True, help="directory containing visible RGB images")
    parser.add_argument("--swir", required=True, help="directory containing aligned SWIR images")
    parser.add_argument("--lwir", required=True, help="directory containing aligned LWIR images")
    parser.add_argument("--output", default="results", help="output directory")
    parser.add_argument("--checkpoint", default="checkpoints/rotfuse_seus.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    visible, swir, lwir = image_map(args.visible), image_map(args.swir), image_map(args.lwir)
    names = sorted(visible.keys() & swir.keys() & lwir.keys())
    if not names:
        raise ValueError("The three folders contain no images with matching filename stems.")
    unmatched = (set(visible) | set(swir) | set(lwir)) - set(names)
    if unmatched:
        print(f"Warning: ignoring {len(unmatched)} filename stem(s) not present in all three folders.")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = RotFuse().to(device).eval()
    load_checkpoint(model, args.checkpoint, device)
    print(f"Loaded {args.checkpoint} on {device}; processing {len(names)} triplet(s).")

    with torch.inference_mode():
        for index, name in enumerate(names, start=1):
            rgb = load_rgb(visible[name]).to(device)
            swir_image = load_gray(swir[name]).to(device)
            lwir_image = load_gray(lwir[name]).to(device)
            spatial_shapes = {tuple(x.shape[-2:]) for x in (rgb, swir_image, lwir_image)}
            if len(spatial_shapes) != 1:
                raise ValueError(f"Input size mismatch for '{name}': {sorted(spatial_shapes)}")

            y, cb, cr = rgb_to_ycbcr(rgb)
            fused_y = model(y, swir_image, lwir_image)
            fused_rgb = ycbcr_to_rgb(fused_y, cb, cr)[0].cpu()
            array = (fused_rgb.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            Image.fromarray(array).save(output / f"{name}.png")
            print(f"[{index}/{len(names)}] {name}.png")


if __name__ == "__main__":
    main()
