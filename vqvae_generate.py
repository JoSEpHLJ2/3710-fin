# vqvae_generate.py
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataset import HipMRISliceDataset
from modules import VQVAE2
from PIL import Image
import torchvision.transforms as transforms

# -----------------------
# 配置
# -----------------------
data_root = "processed_slices"           # 输入PNG图像根目录
model_path = "outputs/vqvae_model.pth"  # 已训练好的模型
output_gen_dir = Path("outputs/generated")
output_cmp_dir = Path("outputs/comparison")  # 可选：原图+生成图对比
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------
# 数据集和DataLoader
# -----------------------
dataset = HipMRISliceDataset(data_root, mode='img', transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# -----------------------
# 加载模型
# -----------------------
model = VQVAE2().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -----------------------
# 遍历生成
# -----------------------
for idx, img_tensor in enumerate(dataloader):
    img_tensor = img_tensor.to(device)  # [B, C, H, W]
    with torch.no_grad():
        recon, _, _ = model(img_tensor)

    # 转为 PIL Image
    recon_img = recon[0,0].cpu().mul(255).byte().numpy()
    recon_pil = Image.fromarray(recon_img)

    # 获取原图路径信息
    orig_path = Path(dataset.items[idx][1])
    case_folder = orig_path.parent.name
    # 保存生成图
    case_output_dir = output_gen_dir / case_folder
    case_output_dir.mkdir(parents=True, exist_ok=True)
    recon_pil.save(case_output_dir / orig_path.name)

    # 可选：保存原图+生成图对比图
    cmp_dir = output_cmp_dir / case_folder
    cmp_dir.mkdir(parents=True, exist_ok=True)
    orig_img = Image.open(orig_path).convert("L")
    cmp_image = Image.new("L", (orig_img.width * 2, orig_img.height))
    cmp_image.paste(orig_img, (0,0))
    cmp_image.paste(recon_pil, (orig_img.width,0))
    cmp_image.save(cmp_dir / orig_path.name)

    if idx % 100 == 0:
        print(f"Processed {idx+1}/{len(dataset)} images")

print("✅ Generation finished. Generated images in 'outputs/generated/' and comparisons in 'outputs/comparison/'")
