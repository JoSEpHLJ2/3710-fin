# train.py
import os
import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from modules import VQVAE2
from dataset import HipMRISliceDataset
from utils import compute_batch_ssim, save_reconstructions
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def get_data_loaders(data_dir, mode, batch_size, val_split=0.1):
    dataset = HipMRISliceDataset(data_dir, mode=mode)
    n = len(dataset)
    nval = max( max(1, int(n * val_split)), 1)
    ntrain = n - nval
    train_set, val_set = random_split(dataset, [ntrain, nval])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, dataset

def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, full_dataset = get_data_loaders(args.data_dir, args.mode, args.batch_size, args.val_split)
    model = VQVAE2(in_ch=1, hidden=args.hidden, num_embed_top=args.num_embed_top, num_embed_bottom=args.num_embed_bottom, commitment_cost=args.commitment).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=args.logdir)
    best_ssim = 0.0
    early_stop_counter = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} Train")
        for batch in pbar:
            imgs = batch.to(device)
            recon, vq_loss, _ = model(imgs)
            recon_loss = torch.mean((recon - imgs) ** 2)
            loss = recon_loss + vq_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        # validation
        model.eval()
        val_loss = 0.0
        ss_total = 0.0
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch.to(device)
                recon, vq_loss, _ = model(imgs)
                recon_loss = torch.mean((recon - imgs) ** 2)
                loss = recon_loss + vq_loss
                val_loss += loss.item()
                ss = compute_batch_ssim(imgs, recon)
                ss_total += ss * imgs.size(0)
                count += imgs.size(0)
        avg_val_loss = val_loss / len(val_loader)
        mean_ssim = ss_total / (count if count>0 else 1)

        print(f"Epoch {epoch} TrainLoss {avg_train_loss:.4f} ValLoss {avg_val_loss:.4f} ValSSIM {mean_ssim:.4f}")
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Metrics/Val_SSIM", mean_ssim, epoch)

        # save sample reconstructions (first batch)
        sample_batch = next(iter(val_loader))
        sample_batch = sample_batch.to(device)
        recon_sample, _, _ = model(sample_batch)
        save_reconstructions(sample_batch, recon_sample, args.output_dir, prefix=f"epoch{epoch}", nrow=min(8, sample_batch.size(0)))

        # checkpoint if best
        if mean_ssim > best_ssim:
            best_ssim = mean_ssim
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_vqvae2.pth"))
            print(f"Saved best model (SSIM={best_ssim:.4f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # early stopping
        if args.early_stop and early_stop_counter >= args.patience:
            print("Early stopping triggered.")
            break

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--mode", type=str, default="nifti", choices=['nifti','img'])
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num_embed_top", type=int, default=512)
    parser.add_argument("--num_embed_bottom", type=int, default=512)
    parser.add_argument("--commitment", type=float, default=0.25)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_loop(args)
