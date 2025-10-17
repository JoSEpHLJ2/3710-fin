# vqvae2_train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from modules import VQVAE2  # 请确保 modules.py 中有 VQVAE2 类

# -----------------------
# Dataset
# -----------------------
class MRI2DDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        for patient in os.listdir(root_dir):
            patient_dir = os.path.join(root_dir, patient)
            if os.path.isdir(patient_dir):
                for f in os.listdir(patient_dir):
                    if f.endswith(".png"):
                        self.images.append(os.path.join(patient_dir, f))
        print(f"Found {len(self.images)} images.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img

# -----------------------
# Train function
# -----------------------
def train(model, dataloader, device, epochs=50, lr=1e-4, save_dir="outputs"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    os.makedirs(save_dir, exist_ok=True)
    loss_list = []

    for epoch in range(epochs):
        running_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs in pbar:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon, vq_loss, _ = model(imgs)
            recon_loss = criterion(recon, imgs)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        loss_list.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

        # 保存每轮的重建样本（前 4 张）
        recon_imgs = recon.detach()[:4]
        for i in range(recon_imgs.size(0)):
            plt.imsave(os.path.join(save_dir, f"recon_epoch{epoch+1}_{i}.png"),
                       recon_imgs[i].cpu().numpy().squeeze(), cmap='gray')

    # 保存 loss 曲线
    plt.figure()
    plt.plot(loss_list, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    print(f"✅ Loss curve saved to {save_dir}/loss_curve.png")
    return model

# -----------------------
# 生成样本函数
# -----------------------
def generate_samples(model, dataloader, device, save_dir="outputs/generated", n_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, imgs in enumerate(dataloader):
            imgs = imgs.to(device)
            recon, _, _ = model(imgs)
            for j in range(min(n_samples, recon.size(0))):
                orig = imgs[j].cpu().numpy().squeeze()
                recon_img = recon[j].cpu().numpy().squeeze()
                plt.imsave(os.path.join(save_dir, f"orig_{i}_{j}.png"), orig, cmap='gray')
                plt.imsave(os.path.join(save_dir, f"recon_{i}_{j}.png"), recon_img, cmap='gray')
                s = ssim(orig, recon_img, data_range=1.0)
                print(f"Image {i}_{j} SSIM: {s:.4f}")
            if i >= n_samples-1:
                break
    print(f"✅ Samples saved to {save_dir}/")

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="processed_slices")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num_embed_top", type=int, default=512)
    parser.add_argument("--num_embed_bottom", type=int, default=512)
    parser.add_argument("--commitment", type=float, default=0.25)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--model_path", type=str, default="outputs/vqvae2_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = MRI2DDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = VQVAE2(in_ch=1, hidden=args.hidden,
                   num_embed_top=args.num_embed_top,
                   num_embed_bottom=args.num_embed_bottom,
                   commitment_cost=args.commitment).to(device)

    if args.generate:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        generate_samples(model, dataloader, device)
    else:
        trained_model = train(model, dataloader, device,
                              epochs=args.epochs, lr=args.lr)
        torch.save(trained_model.state_dict(), args.model_path)
        print(f"✅ Model saved to {args.model_path}")

if __name__ == "__main__":
    main()
