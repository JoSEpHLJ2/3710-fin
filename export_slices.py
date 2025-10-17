# export_slices.py
import os
import glob
import nibabel as nib
import numpy as np
from PIL import Image
import argparse

def export_all(nifti_dir, out_dir, axis=2, resize=None, max_slices_per_nifti=None):
    os.makedirs(out_dir, exist_ok=True)
    files = glob.glob(os.path.join(nifti_dir, '**/*.nii'), recursive=True) + \
            glob.glob(os.path.join(nifti_dir, '**/*.nii.gz'), recursive=True)
    for f in files:
        try:
            img = nib.load(f)
            arr = img.get_fdata()
            mn, mx = arr.min(), arr.max()
            if mx>mn:
                arr = (arr - mn) / (mx - mn)
            else:
                arr = np.zeros_like(arr)
            # create patient folder
            base = os.path.basename(f).replace('.nii.gz','').replace('.nii','')
            od = os.path.join(out_dir, base)
            os.makedirs(od, exist_ok=True)
            count = 0
            for i in range(arr.shape[axis]):
                if axis==0:
                    sl = arr[i,:,:]
                elif axis==1:
                    sl = arr[:,i,:]
                else:
                    sl = arr[:,:,i]
                if sl.std() < 1e-6:
                    continue
                img8 = (sl * 255.0).astype(np.uint8)
                pil = Image.fromarray(img8)
                if resize:
                    pil = pil.resize(resize)
                pil.save(os.path.join(od, f"{base}_slice{i:03d}.png"))
                count += 1
                if max_slices_per_nifti and count >= max_slices_per_nifti:
                    break
            print(f"Exported {count} slices from {f}")
        except Exception as e:
            print(f"Failed {f}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--axis", type=int, default=2)
    parser.add_argument("--resize", type=int, nargs=2, default=None)
    parser.add_argument("--max_slices", type=int, default=None)
    args = parser.parse_args()
    export_all(args.nifti_dir, args.out_dir, axis=args.axis, resize=tuple(args.resize) if args.resize else None, max_slices_per_nifti=args.max_slices)
# export_slices.py
import os
import glob
import nibabel as nib
import numpy as np
from PIL import Image
import argparse

def export_all(nifti_dir, out_dir, axis=2, resize=None, max_slices_per_nifti=None):
    os.makedirs(out_dir, exist_ok=True)
    files = glob.glob(os.path.join(nifti_dir, '**/*.nii'), recursive=True) + \
            glob.glob(os.path.join(nifti_dir, '**/*.nii.gz'), recursive=True)
    for f in files:
        try:
            img = nib.load(f)
            arr = img.get_fdata()
            mn, mx = arr.min(), arr.max()
            if mx>mn:
                arr = (arr - mn) / (mx - mn)
            else:
                arr = np.zeros_like(arr)
            # create patient folder
            base = os.path.basename(f).replace('.nii.gz','').replace('.nii','')
            od = os.path.join(out_dir, base)
            os.makedirs(od, exist_ok=True)
            count = 0
            for i in range(arr.shape[axis]):
                if axis==0:
                    sl = arr[i,:,:]
                elif axis==1:
                    sl = arr[:,i,:]
                else:
                    sl = arr[:,:,i]
                if sl.std() < 1e-6:
                    continue
                img8 = (sl * 255.0).astype(np.uint8)
                pil = Image.fromarray(img8)
                if resize:
                    pil = pil.resize(resize)
                pil.save(os.path.join(od, f"{base}_slice{i:03d}.png"))
                count += 1
                if max_slices_per_nifti and count >= max_slices_per_nifti:
                    break
            print(f"Exported {count} slices from {f}")
        except Exception as e:
            print(f"Failed {f}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nifti_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--axis", type=int, default=2)
    parser.add_argument("--resize", type=int, nargs=2, default=None)
    parser.add_argument("--max_slices", type=int, default=None)
    args = parser.parse_args()
    export_all(args.nifti_dir, args.out_dir, axis=args.axis, resize=tuple(args.resize) if args.resize else None, max_slices_per_nifti=args.max_slices)
