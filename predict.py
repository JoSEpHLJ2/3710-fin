# predict.py
import os
import torch
from torch.utils.data import DataLoader
from modules import VQVAE2
from dataset import HipMRISliceDataset
from utils import compute_batch_ssim, save_reconstructions
import argparse
from tqdm import tqdm

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HipMRISliceDataset(args.data_dir, mode=args.mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = VQVAE2(in_ch=1, hidden=args.hidden, num_embed_top=args.num_embed_top, num_embed_bottom=args.num_embed_bottom, commitment_cost=args.commitment).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    ss_total = 0.0
    n = 0
    os.makedirs(args.output_dir, exist_ok=True)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            imgs = batch.to(device)
            recon, _, _ = model(imgs)
            ss = compute_batch_ssim(imgs, recon)
            ss_total += ss * imgs.size(0)
            n += imgs.size(0)
            # save first few batches' reconstructions
            if i < args.save_batches:
                save_reconstructions(imgs, recon, args.output_dir, prefix=f"batch{i}", nrow=min(8, imgs.size(0)))
    mean_ssim = ss_total / max(1, n)
    print(f"Mean SSIM on dataset: {mean_ssim:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--mode", default="nifti", choices=['nifti','img'])
    parser.add_argument("--model_path", default="outputs/best_vqvae2.pth")
    parser.add_argument("--output_dir", default="outputs/pred")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num_embed_top", type=int, default=512)
    parser.add_argument("--num_embed_bottom", type=int, default=512)
    parser.add_argument("--commitment", type=float, default=0.25)
    parser.add_argument("--save_batches", type=int, default=3)
    args = parser.parse_args()
    evaluate(args)
