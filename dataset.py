# dataset.py
import os
import glob
import random
import numpy as np
from PIL import Image
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def nifti_to_slices(nifti_path, axis=2, min_valid_std=1e-6):
    """读取 NIfTI 并返回经过归一化的 2D 切片数组列表（float32, [0,1]）"""
    img = nib.load(nifti_path)
    arr = img.get_fdata()
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)
    slices = []
    for i in range(arr.shape[axis]):
        if axis == 0:
            sl = arr[i,:,:]
        elif axis == 1:
            sl = arr[:,i,:]
        else:
            sl = arr[:,:,i]
        if sl.std() > min_valid_std:
            slices.append(sl.astype(np.float32))
    return slices

class HipMRISliceDataset(Dataset):
    """
    支持两种模式：
      mode='nifti' ：递归读取目录下所有 .nii/.nii.gz 文件并随机抽取切片
      mode='img'   ：读取 png/jpg 图像文件
    输出：torch.Tensor [C, H, W], C=1
    """
    def __init__(self, data_dir, mode='nifti', transform=None, max_slices_per_nifti=100):
        self.items = []
        self.mode = mode
        self.transform = transform
        self.max_slices_per_nifti = max_slices_per_nifti

        if mode == 'nifti':
            files = glob.glob(os.path.join(data_dir, '**/*.nii'), recursive=True) + \
                    glob.glob(os.path.join(data_dir, '**/*.nii.gz'), recursive=True)
            for f in files:
                self.items.append(('nifti', f))
        else:
            files = glob.glob(os.path.join(data_dir, '**/*.png'), recursive=True) + \
                    glob.glob(os.path.join(data_dir, '**/*.jpg'), recursive=True) + \
                    glob.glob(os.path.join(data_dir, '**/*.jpeg'), recursive=True)
            for f in files:
                self.items.append(('img', f))

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256,256)),
                transforms.ToTensor()  # -> [C, H, W] in [0,1]
            ])

    def __len__(self):
        # note: for nifti mode we count number of nifti files (slices chosen at getitem)
        return len(self.items)

    def __getitem__(self, idx):
        mode, path = self.items[idx]
        if mode == 'nifti':
            slices = nifti_to_slices(path)
            if len(slices) == 0:
                # fallback: return zeros
                arr = np.zeros((256,256), dtype=np.float32)
            else:
                # 随机选择一个切片（可修改为顺序或多切片）
                chosen = slices[random.randint(0, len(slices)-1)]
                arr = (chosen * 255.0).astype(np.uint8)
            img = np.expand_dims(arr, axis=2)  # H,W,1
            if self.transform:
                t = self.transform(img)  # [C,H,W]
            else:
                t = torch.from_numpy(arr / 255.).unsqueeze(0).float()
            return t
        else:
            img = Image.open(path).convert('L')
            if self.transform:
                t = self.transform(np.array(img))
            else:
                t = transforms.ToTensor()(img)
            return t
