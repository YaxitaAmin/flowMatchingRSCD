import pyarrow.parquet as pq
from PIL import Image
import io
import os
from tqdm import tqdm

RAW_DIR = "data/raw/LEVIR_CDPlus/data"
OUT_DIR = "data/processed"

SPLITS = {
    "train": [f"train-0000{i}-of-00005.parquet" for i in range(5)],
    "test":  [f"test-0000{i}-of-00003.parquet"  for i in range(3)],
}

def save_image(img_dict, out_path):
    img = Image.open(io.BytesIO(img_dict["bytes"]))
    img.save(out_path)

def extract_split(split_name, parquet_files):
    for folder in ["A", "B", "label"]:
        os.makedirs(os.path.join(OUT_DIR, split_name, folder), exist_ok=True)

    idx = 0
    for pfile in parquet_files:
        fpath = os.path.join(RAW_DIR, pfile)
        print(f"reading {pfile}...")
        table = pq.read_table(fpath).to_pydict()

        for i in tqdm(range(len(table["image1"])), desc=split_name):
            name = f"{idx:05d}.png"
            save_image(table["image1"][i], os.path.join(OUT_DIR, split_name, "A", name))
            save_image(table["image2"][i], os.path.join(OUT_DIR, split_name, "B", name))
            save_image(table["mask"][i],   os.path.join(OUT_DIR, split_name, "label", name))
            idx += 1

    print(f"{split_name} done — {idx} pairs saved!")

if __name__ == "__main__":
    for split, files in SPLITS.items():
        extract_split(split, files)
