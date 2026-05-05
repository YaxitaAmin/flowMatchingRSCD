import os
import sys
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.dataset import LEVIRDataset, SyntheticDataset
from src.models.change_detection import ChangeDetectionUNet
from src.evaluation.metrics import evaluate_batch, aggregate_metrics

def train_config(config, config_name, train_dataset, val_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\ntraining config: {config_name} on {device}")

    train_dl = DataLoader(
        train_dataset,
        batch_size  = config["change_detection"]["batch_size"],
        shuffle     = True,
        num_workers = config["data"]["num_workers"],
        pin_memory  = True
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size  = config["change_detection"]["batch_size"],
        shuffle     = False,
        num_workers = config["data"]["num_workers"],
        pin_memory  = True
    )

    model     = ChangeDetectionUNet(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config["change_detection"]["learning_rate"],
        weight_decay = 1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = config["change_detection"]["epochs"]
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    ckpt_dir = os.path.join(config["paths"]["checkpoints"], config_name)
    log_dir  = config["paths"]["logs"]
    os.makedirs(ckpt_dir, exist_ok=True)

    log_path  = os.path.join(log_dir, f"cd_{config_name}.log")
    best_f1   = 0.0

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_f1,val_iou,val_precision,val_recall\n")

    epochs = config["change_detection"]["epochs"]

    for epoch in range(1, epochs + 1):
        # training
        model.train()
        train_losses = []
        pbar = tqdm(train_dl, desc=f"[{config_name}] epoch {epoch}/{epochs}")

        for img_a, img_b, mask in pbar:
            img_a, img_b, mask = img_a.to(device), img_b.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(img_a, img_b)
            loss   = criterion(logits, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # validation
        model.eval()
        val_metrics = []
        with torch.no_grad():
            for img_a, img_b, mask in val_dl:
                img_a, img_b, mask = img_a.to(device), img_b.to(device), mask.to(device)
                logits = model(img_a, img_b)
                val_metrics.append(evaluate_batch(logits, mask))

        avg_train_loss = np.mean(train_losses)
        avg_val        = aggregate_metrics(val_metrics)

        print(f"epoch {epoch} | train loss: {avg_train_loss:.4f} | "
              f"val f1: {avg_val['f1']:.4f} | val iou: {avg_val['iou']:.4f}")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.6f},{avg_val['f1']:.6f},"
                    f"{avg_val['iou']:.6f},{avg_val['precision']:.6f},{avg_val['recall']:.6f}\n")

        if avg_val["f1"] > best_f1:
            best_f1 = avg_val["f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "f1": best_f1
            }, os.path.join(ckpt_dir, "best.pt"))

    print(f"config {config_name} done! best val f1: {best_f1:.4f}")
    return best_f1

def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # datasets for each config
    real_train    = LEVIRDataset("train_patches", config, augment=False)
    real_train_aug= LEVIRDataset("train_patches", config, augment=True)
    val_ds        = LEVIRDataset("test_patches",  config, augment=False)

    synthetic_ds = SyntheticDataset(config)
    configs = {
        #"config1_baseline":         real_train,
        #"config2_augmented":        real_train_aug,
        #"config3_flow_matching":    ConcatDataset([real_train, synthetic_ds]),
        "config4_full_pipeline":    ConcatDataset([real_train_aug, synthetic_ds]),
    }

    results = {}
    for name, train_ds in configs.items():
        results[name] = train_config(config, name, train_ds, val_ds)

    print("\nall configs done!")
    for name, f1 in results.items():
        print(f"  {name}: best f1 = {f1:.4f}")

if __name__ == "__main__":
    main()
