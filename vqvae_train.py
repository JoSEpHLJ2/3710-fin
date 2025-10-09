# vqvae_train.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

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
# VQ-VAE Model
# -----------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, latent_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 4, 2, 1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 4, 2, 1)
        self.conv3 = nn.Conv2d(hidden_channels, latent_dim, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_channels=64, out_channels=1):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_dim, hidden_channels, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, 2, 1)
        self.deconv3 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.sigmoid(self.deconv3(x))
        return x

class VQVAE(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_channels, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_channels, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# -----------------------
# Train Function
# -----------------------
def train(model, dataloader, device, epochs=50, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    loss_list = []

    for epoch in range(epochs):
        running_loss = 0
        for imgs in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon = model(imgs)
            loss = criterion(recon, imgs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        loss_list.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    # 绘制 loss 曲线
    plt.figure()
    plt.plot(loss_list, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig("outputs/loss_curve.png")
    print("✅ Loss curve saved to outputs/loss_curve.png")
    return model

# -----------------------
# Generate Samples
# -----------------------
def generate_samples(model, dataloader, device, n_samples=5):
    os.makedirs("outputs/generated", exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, imgs in enumerate(dataloader):
            imgs = imgs.to(device)
            recon = model(imgs)
            for j in range(min(n_samples, recon.size(0))):
                orig = imgs[j].cpu().numpy().squeeze()
                recon_img = recon[j].cpu().numpy().squeeze()
                # 保存原图和重建图
                plt.imsave(f"outputs/generated/orig_{i}_{j}.png", orig, cmap='gray')
                plt.imsave(f"outputs/generated/recon_{i}_{j}.png", recon_img, cmap='gray')
                # 计算 SSIM
                s = ssim(orig, recon_img)
                print(f"Image {i}_{j} SSIM: {s:.4f}")
            if i >= n_samples-1:
                break
    print("✅ Samples generated in outputs/generated/")

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="processed_slices")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--model_path", type=str, default="outputs/vqvae_model.pth")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = MRI2DDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = VQVAE().to(device)

    if args.generate:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        generate_samples(model, dataloader, device)
    else:
        trained_model = train(model, dataloader, device, epochs=args.epochs, lr=args.lr)
        torch.save(trained_model.state_dict(), args.model_path)
        print(f"✅ Model saved to {args.model_path}")

if __name__ == "__main__":
    main()
