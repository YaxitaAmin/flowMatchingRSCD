import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# settings
PROCESSED_DIR = "data/processed"
PATCH_SIZE = 256
CHANGE_RATIO_MIN = 0.01  # filter patches with less than 1% change pixels
SPLITS = ["train", "test"]

def slice_into_patches(img, patch_size):
    """slice a PIL image into non-overlapping patches."""
    w, h = img.size
    patches = []
    for top in range(0, h, patch_size):
        for left in range(0, w, patch_size):
            box = (left, top, left + patch_size, top + patch_size)
            patch = img.crop(box)
            if patch.size == (patch_size, patch_size):
                patches.append(patch)
    return patches

def compute_change_ratio(mask_patch):
    """compute fraction of changed pixels in a mask patch."""
    arr = np.array(mask_patch)
    return (arr > 0).sum() / arr.size

def process_split(split):
    a_dir     = os.path.join(PROCESSED_DIR, split, "A")
    b_dir     = os.path.join(PROCESSED_DIR, split, "B")
    label_dir = os.path.join(PROCESSED_DIR, split, "label")

    out_a     = os.path.join(PROCESSED_DIR, split + "_patches", "A")
    out_b     = os.path.join(PROCESSED_DIR, split + "_patches", "B")
    out_label = os.path.join(PROCESSED_DIR, split + "_patches", "label")

    for d in [out_a, out_b, out_label]:
        os.makedirs(d, exist_ok=True)

    image_names = sorted(os.listdir(a_dir))
    patch_idx = 0
    total_kept = 0
    total_dropped = 0

    for name in tqdm(image_names, desc=f"patching {split}"):
        img_a = Image.open(os.path.join(a_dir, name))
        img_b = Image.open(os.path.join(b_dir, name))
        mask  = Image.open(os.path.join(label_dir, name))

        patches_a    = slice_into_patches(img_a, PATCH_SIZE)
        patches_b    = slice_into_patches(img_b, PATCH_SIZE)
        patches_mask = slice_into_patches(mask,  PATCH_SIZE)

        for pa, pb, pm in zip(patches_a, patches_b, patches_mask):
            ratio = compute_change_ratio(pm)
            if ratio < CHANGE_RATIO_MIN:
                total_dropped += 1
                continue

            fname = f"{patch_idx:06d}.png"
            pa.save(os.path.join(out_a,     fname))
            pb.save(os.path.join(out_b,     fname))
            pm.save(os.path.join(out_label, fname))
            patch_idx += 1
            total_kept += 1

    print(f"{split} — kept: {total_kept}, dropped: {total_dropped}")

if __name__ == "__main__":
    for split in SPLITS:
        process_split(split)
    print("all done!")
