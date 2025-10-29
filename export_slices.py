"""Batch export of 2D slices from a directory of NIfTI volumes."""

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


def export_slices(
    nifti_path: Path,
    out_dir: Path,
    axis: int,
    resize: tuple[int, int] | None,
    max_slices: int | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(nifti_path))
    arr = img.get_fdata().astype(np.float32)
    mn, mx = arr.min(), arr.max()
    arr = (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)

    count = 0
    for i in range(arr.shape[axis]):
        if axis == 0:
            sl = arr[i, :, :]
        elif axis == 1:
            sl = arr[:, i, :]
        else:
            sl = arr[:, :, i]
        if sl.std() < 1e-6:
            continue
        img8 = (sl * 255.0).astype(np.uint8)
        pil = Image.fromarray(img8)
        if resize:
            pil = pil.resize(resize)
        pil.save(out_dir / f"slice_{i:03d}.png")
        count += 1
        if max_slices is not None and count >= max_slices:
            break



def export_all(
    nifti_dir: Path,
    out_dir: Path,
    axis: int,
    resize: tuple[int, int] | None,
    max_slices: int | None,
) -> None:
    nifti_files = sorted(nifti_dir.rglob("*.nii")) + sorted(nifti_dir.rglob("*.nii.gz"))
    for f in nifti_files:
        subdir = out_dir / f.stem.replace(".nii", "")
        print(f"Exporting {f} -> {subdir}")
        export_slices(f, subdir, axis, resize, max_slices)


+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(description=__doc__)
+    parser.add_argument("--nifti_dir", type=Path, required=True)
+    parser.add_argument("--out_dir", type=Path, required=True)
+    parser.add_argument("--axis", type=int, default=2)
+    parser.add_argument("--resize", type=int, nargs=2, default=None)
+    parser.add_argument("--max_slices", type=int, default=None)
+    return parser.parse_args()
+
 
 def main() -> None:
-    parser = argparse.ArgumentParser(description=__doc__)
-    parser.add_argument("--nifti_dir", type=Path, required=True)
-    parser.add_argument("--out_dir", type=Path, required=True)
-    parser.add_argument("--axis", type=int, default=2)
-    parser.add_argument("--resize", type=int, nargs=2, default=None)
-    parser.add_argument("--max_slices", type=int, default=None)
-    args = parser.parse_args()
-    resize = tuple(args.resize) if args.resize else None
-    export_all(args.nifti_dir, args.out_dir, args.axis, resize, args.max_slices)
+    args = parse_args()
+    resize = tuple(args.resize) if args.resize else None
+    export_all(args.nifti_dir, args.out_dir, args.axis, resize, args.max_slices)
 
 
 if __name__ == "__main__":
     main()
