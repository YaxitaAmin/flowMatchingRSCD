import argparse
import os
import sys
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.dataset import LEVIRDataset
from src.models.flow_matching import FlowMatching

def train(config_path="configs/config.yaml", resume=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    train_ds = LEVIRDataset("train_patches", config, augment=False)
    train_dl = DataLoader(
        train_ds,
        batch_size  = config["flow_matching"]["batch_size"],
        shuffle     = True,
        num_workers = config["data"]["num_workers"],
        pin_memory  = True
    )
    print(f"training samples: {len(train_ds)}")

    model     = FlowMatching(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config["flow_matching"]["learning_rate"],
        weight_decay = 1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = config["flow_matching"]["epochs"]
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {total_params:,}")

    ckpt_dir = config["paths"]["checkpoints"]
    log_dir  = config["paths"]["logs"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    log_path  = os.path.join(log_dir, "flow_matching_train.log")
    best_loss = float("inf")

    with open(log_path, "w") as log_file:
        log_file.write("epoch,avg_loss,lr\n")

    epochs = config["flow_matching"]["epochs"]
    start_epoch = 1

    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"resumed from epoch {ckpt['epoch']}, continuing from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_losses = []

        pbar = tqdm(train_dl, desc=f"epoch {epoch}/{epochs}")
        for img_a, img_b, _ in pbar:
            x1 = torch.cat([img_a, img_b], dim=1).to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(x1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss   = np.mean(epoch_losses)
        current_lr = scheduler.get_last_lr()[0]
        print(f"epoch {epoch} avg loss: {avg_loss:.4f} lr: {current_lr:.6f}")

        with open(log_path, "a") as log_file:
            log_file.write(f"{epoch},{avg_loss:.6f},{current_lr:.6f}\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss
            }, os.path.join(ckpt_dir, "flow_matching_best.pt"))

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            }, os.path.join(ckpt_dir, f"flow_matching_epoch{epoch}.pt"))

    print(f"training complete! best loss: {best_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train(resume=args.resume)
