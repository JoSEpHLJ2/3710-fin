import os
import nibabel as nib
import numpy as np
from PIL import Image

# 输入和输出目录
input_dir = "data/HipMRI_study_complete_release_v1/semantic_MRs_anon"
output_dir = "processed_slices"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 遍历每个 NIfTI 文件
nii_files = [f for f in os.listdir(input_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]

print(f"找到 {len(nii_files)} 个 NIfTI 文件，将进行转换...")

for nii_file in nii_files:
    nii_path = os.path.join(input_dir, nii_file)
    
    # 读取 NIfTI 文件
    nii_img = nib.load(nii_path)
    img_data = nii_img.get_fdata()  # 形状一般是 (H, W, D)
    
    # 获取病人名称，去掉扩展名
    patient_name = os.path.splitext(os.path.splitext(nii_file)[0])[0]  # 去掉 .nii.gz
    patient_dir = os.path.join(output_dir, patient_name)
    os.makedirs(patient_dir, exist_ok=True)
    
    # 遍历每一张切片并保存为 PNG
    for i in range(img_data.shape[2]):  # 假设第三维是切片
        slice_img = img_data[:, :, i]
        
        # 归一化到 0-255
        slice_min, slice_max = slice_img.min(), slice_img.max()
        if slice_max > slice_min:
            slice_norm = ((slice_img - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
        else:
            slice_norm = np.zeros_like(slice_img, dtype=np.uint8)
        
        slice_pil = Image.fromarray(slice_norm)
        slice_pil.save(os.path.join(patient_dir, f"slice_{i:03d}.png"))

print("✅ 转换完成！PNG 图片已保存到 processed_slices/")
