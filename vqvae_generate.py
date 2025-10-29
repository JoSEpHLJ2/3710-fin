"""Generate reconstructions using a trained VQ-VAE2 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import HipMRISliceDataset
from modules import VQVAE2
from utils import save_reconstructions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--mode", choices=["nifti", "img"], default="nifti")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/generate"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num_embed_top", type=int, default=512)
    parser.add_argument("--num_embed_bottom", type=int, default=512)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--max_slices_per_nifti", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HipMRISliceDataset(
        str(args.data_dir),
        mode=args.mode,
        max_slices_per_nifti=None if args.max_slices_per_nifti <= 0 else args.max_slices_per_nifti,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = VQVAE2(
        in_ch=1,
        hidden=args.hidden,
        num_embed_top=args.num_embed_top,
        num_embed_bottom=args.num_embed_bottom,
        commitment_cost=args.commitment_cost,
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, imgs in enumerate(loader):
            imgs = imgs.to(device)
            recon, _, _ = model(imgs)
            save_reconstructions(imgs.cpu(), recon.cpu(), str(args.output_dir), prefix=f"batch{batch_idx}")
            if batch_idx >= 4:  # 导出前几个 batch 即可
                break
    print(f"Generated samples saved to {args.output_dir}")


if __name__ == "__main__":
    main()
