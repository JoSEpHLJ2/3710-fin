# vqvae_train.py
"""
Standalone training script for the hierarchical VQ-VAE2 model on HipMRI slices.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HipMRISliceDataset
from modules import VQVAE2
from utils import compute_batch_ssim, save_reconstructions


def build_dataloader(
    data_dir: str,
    mode: str,
    batch_size: int,
    num_workers: int,
    max_slices_per_nifti: int | None,
) -> DataLoader:
    dataset = HipMRISliceDataset(
        data_dir,
        mode=mode,
        max_slices_per_nifti=max_slices_per_nifti,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    max_slices = None if args.max_slices_per_nifti <= 0 else args.max_slices_per_nifti
    dataloader = build_dataloader(
        args.data_dir,
        args.mode,
        args.batch_size,
        args.num_workers,
        max_slices,
    )

    model = VQVAE2(
        in_ch=1,
        hidden=args.hidden,
        num_embed_top=args.num_embed_top,
        num_embed_bottom=args.num_embed_bottom,
        commitment_cost=args.commitment_cost,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_ssim = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_ssim = 0.0
        batch_count = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            imgs = batch.to(device)
            recon, vq_loss, _ = model(imgs)
            recon_loss = F.mse_loss(recon, imgs)
            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_ssim = compute_batch_ssim(imgs, recon)
            total_ssim += batch_ssim
            batch_count += 1
            pbar.set_postfix(
                loss=loss.item(),
                recon=recon_loss.item(),
                ssim=batch_ssim,
            )

        avg_loss = total_loss / max(1, batch_count)
        avg_ssim = total_ssim / max(1, batch_count)
        print(f"Epoch {epoch}: loss={avg_loss:.4f} | ssim={avg_ssim:.4f}")

        # 保存可视化
        sample_dir = Path(args.output_dir) / "samples"
        save_reconstructions(imgs.detach().cpu(), recon.detach().cpu(), str(sample_dir), prefix=f"epoch{epoch}")

        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            ckpt_path = Path(args.output_dir) / "best_vqvae2.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved new best model to {ckpt_path} (SSIM={best_ssim:.4f})")

    # 保存最后一次模型
    last_ckpt = Path(args.output_dir) / "last_vqvae2.pth"
    torch.save(model.state_dict(), last_ckpt)
    print(f"Training finished. Final model saved to {last_ckpt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--mode", type=str, default="nifti", choices=["nifti", "img"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num_embed_top", type=int, default=512)
    parser.add_argument("--num_embed_bottom", type=int, default=512)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_slices_per_nifti",
        type=int,
        default=100,
        help="Limit number of slices sampled per NIfTI (<=0 keeps all slices).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
