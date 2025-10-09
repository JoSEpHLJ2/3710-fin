# utils.py
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
from torchvision.utils import save_image

def compute_batch_ssim(x, y):
    """
    x, y: torch tensors [B, C, H, W] in [0,1]
    返回 batch 平均 SSIM
    """
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    ss = []
    for i in range(x_np.shape[0]):
        a = x_np[i].transpose(1,2,0)
        b = y_np[i].transpose(1,2,0)
        if a.shape[2] == 1:
            a = a[:,:,0]
            b = b[:,:,0]
            s = ssim(a, b, data_range=1.0)
        else:
            s = ssim(a, b, data_range=1.0, multichannel=True)
        ss.append(s)
    return float(np.mean(ss))

def save_reconstructions(x, recon, out_dir, prefix='recon', nrow=4):
    """
    保存真实图像与重建并列的图像
    x, recon: [B, C, H, W]
    """
    os.makedirs(out_dir, exist_ok=True)
    B = x.size(0)
    # make pairwise: stack real then recon
    comp = torch.cat([x, recon], dim=0)
    # Normalize (should already be in [0,1])
    save_image(comp, os.path.join(out_dir, f"{prefix}.png"), nrow=nrow)
