import os
import numpy as np
from PIL import Image
from tqdm import tqdm

PATCH_DIR = "data/processed/train_patches"

def compute_mean_std(folder):
    files = sorted(os.listdir(folder))
    pixel_sum    = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count  = 0

    for fname in tqdm(files, desc=f"computing stats for {os.path.basename(folder)}"):
        img = np.array(Image.open(os.path.join(folder, fname))).astype(np.float64) / 255.0
        pixel_sum    += img.reshape(-1, 3).sum(axis=0)
        pixel_sq_sum += (img ** 2).reshape(-1, 3).sum(axis=0)
        pixel_count  += img.shape[0] * img.shape[1]

    mean = pixel_sum / pixel_count
    std  = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)
    return mean, std

if __name__ == "__main__":
    mean_a, std_a = compute_mean_std(os.path.join(PATCH_DIR, "A"))
    mean_b, std_b = compute_mean_std(os.path.join(PATCH_DIR, "B"))

    # combined stats across both time points
    mean_combined = (mean_a + mean_b) / 2
    std_combined  = (std_a  + std_b)  / 2

    print(f"\nt1 mean: {mean_a.tolist()}")
    print(f"t1 std:  {std_a.tolist()}")
    print(f"\nt2 mean: {mean_b.tolist()}")
    print(f"t2 std:  {std_b.tolist()}")
    print(f"\ncombined mean: {mean_combined.tolist()}")
    print(f"combined std:  {std_combined.tolist()}")

    # save to file
    np.save("data/processed/norm_stats.npy", {
        "mean_a": mean_a, "std_a": std_a,
        "mean_b": mean_b, "std_b": std_b,
        "mean_combined": mean_combined,
        "std_combined":  std_combined
    })
    print("\nstats saved to data/processed/norm_stats.npy")
