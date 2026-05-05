import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.flow_matching import FlowMatching

def generate_pairs(config, n_pairs=500, steps=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    ckpt_path = os.path.join(config["paths"]["checkpoints"], "flow_matching_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = FlowMatching(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded flow matching model from epoch {ckpt['epoch']}")

    out_dir = config["paths"]["synthetic_pairs"]
    a_dir = os.path.join(out_dir, "A")
    b_dir = os.path.join(out_dir, "B")
    m_dir = os.path.join(out_dir, "label")
    for d in [a_dir, b_dir, m_dir]:
        os.makedirs(d, exist_ok=True)

    patch_size = config["data"]["patch_size"]
    batch_size = 8
    generated  = 0

    print(f"generating {n_pairs} synthetic pairs...")

    with torch.no_grad():
        for _ in tqdm(range((n_pairs + batch_size - 1) // batch_size)):
            bs = min(batch_size, n_pairs - generated)
            if bs <= 0:
                break

            # use the model's built-in sample method
            x = model.sample((bs, 6, patch_size, patch_size), device=device, steps=steps)

            img_a = x[:, :3, :, :]
            img_b = x[:, 3:, :, :]
            diff  = (img_a - img_b).abs().mean(dim=1, keepdim=True)
            mask  = (diff > diff.mean() + diff.std()).float()

            for j in range(bs):
                idx = generated + j

                def to_uint8(t):
                    arr = t.cpu().numpy().transpose(1, 2, 0)
                    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                    return (arr * 255).astype(np.uint8)

                Image.fromarray(to_uint8(img_a[j])).save(os.path.join(a_dir, f"syn_{idx:05d}.png"))
                Image.fromarray(to_uint8(img_b[j])).save(os.path.join(b_dir, f"syn_{idx:05d}.png"))
                mask_np = (mask[j, 0].cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(mask_np).save(os.path.join(m_dir, f"syn_{idx:05d}.png"))

            generated += bs

    print(f"done! {generated} synthetic pairs saved to {out_dir}")

def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    generate_pairs(config, n_pairs=500)

if __name__ == "__main__":
    main()
