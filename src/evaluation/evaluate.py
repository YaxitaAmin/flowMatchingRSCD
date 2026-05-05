import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.dataset import LEVIRDataset
from src.models.change_detection import ChangeDetectionUNet
from src.evaluation.metrics import (
    threshold_predictions, compute_f1, compute_iou,
    compute_precision, compute_recall, compute_confusion_matrix,
    evaluate_batch, aggregate_metrics
)

def load_model(config, ckpt_path):
    model = ChangeDetectionUNet(config)
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    return model

def evaluate_model(model, dataloader, device):
    model.eval()
    all_metrics = []
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for img_a, img_b, mask in tqdm(dataloader, desc="evaluating"):
            img_a, img_b, mask = img_a.to(device), img_b.to(device), mask.to(device)
            logits = model(img_a, img_b)
            preds  = threshold_predictions(logits)

            all_metrics.append(evaluate_batch(logits, mask))
            all_preds.append(preds.cpu())
            all_targets.append(mask.cpu())

    all_preds   = torch.cat(all_preds,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    avg_metrics = aggregate_metrics(all_metrics)
    cm          = compute_confusion_matrix(all_preds, all_targets)

    return avg_metrics, cm

def plot_confusion_matrix(cm, config_name, save_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["no change", "change"])
    ax.set_yticklabels(["no change", "change"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"confusion matrix: {config_name}")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    path = os.path.join(save_dir, f"cm_{config_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved confusion matrix: {path}")

def plot_all_confusion_matrices(cms_dict, save_dir):
    """side by side confusion matrices for all configs."""
    n = len(cms_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    for ax, (name, cm) in zip(axes, cms_dict.items()):
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["no change", "change"], fontsize=8)
        ax.set_yticklabels(["no change", "change"], fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(name, fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=8)

    plt.suptitle("confusion matrices across all configs", fontsize=12)
    plt.tight_layout()
    path = os.path.join(save_dir, "cm_all_configs.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved combined confusion matrix: {path}")

def plot_metrics_comparison(results_dict, save_dir):
    """bar chart comparing f1 and iou across all configs."""
    configs = list(results_dict.keys())
    f1s     = [results_dict[c]["f1"]  for c in configs]
    ious    = [results_dict[c]["iou"] for c in configs]

    x = np.arange(len(configs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, f1s,  w, label="f1 score",  color="steelblue")
    ax.bar(x + w/2, ious, w, label="iou score", color="darkorange")

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("f1 and iou comparison across configs")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(save_dir, "metrics_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved metrics comparison: {path}")

def run_perturbation_test(model, dataloader, device, perturbation, std=0.05):
    """evaluate model under gaussian noise perturbation."""
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for img_a, img_b, mask in tqdm(dataloader, desc=f"perturb: {perturbation}"):
            img_a, img_b, mask = img_a.to(device), img_b.to(device), mask.to(device)

            if perturbation == "gaussian_noise":
                img_a = img_a + torch.randn_like(img_a) * std
                img_b = img_b + torch.randn_like(img_b) * std
            elif perturbation == "blur":
                import torchvision.transforms.functional as TF
                img_a = TF.gaussian_blur(img_a, kernel_size=5)
                img_b = TF.gaussian_blur(img_b, kernel_size=5)
            elif perturbation == "occlusion":
                h, w  = img_a.shape[2], img_a.shape[3]
                oc    = int(h * 0.1)
                img_a[:, :, :oc, :oc] = 0
                img_b[:, :, :oc, :oc] = 0

            logits = model(img_a, img_b)
            all_metrics.append(evaluate_batch(logits, mask))

    return aggregate_metrics(all_metrics)

def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plot_dir = config["paths"]["plots"]
    ckpt_dir = config["paths"]["checkpoints"]
    os.makedirs(plot_dir, exist_ok=True)

    test_ds = LEVIRDataset("test_patches", config, augment=False)
    test_dl = DataLoader(
        test_ds,
        batch_size  = config["change_detection"]["batch_size"],
        shuffle     = False,
        num_workers = config["data"]["num_workers"],
        pin_memory  = True
    )

    # evaluate all available configs
    config_names = [
        "config1_baseline",
        "config2_augmented",
        "config3_flow_matching",
        "config4_full_pipeline"
    ]

    results_dict = {}
    cms_dict     = {}

    for name in config_names:
        ckpt_path = os.path.join(ckpt_dir, name, "best.pt")
        if not os.path.exists(ckpt_path):
            print(f"skipping {name} — checkpoint not found")
            continue

        print(f"\nevaluating {name}...")
        model = load_model(config, ckpt_path).to(device)
        metrics, cm = evaluate_model(model, test_dl, device)

        results_dict[name] = metrics
        cms_dict[name]     = cm

        print(f"  f1:        {metrics['f1']:.4f}")
        print(f"  iou:       {metrics['iou']:.4f}")
        print(f"  precision: {metrics['precision']:.4f}")
        print(f"  recall:    {metrics['recall']:.4f}")

        plot_confusion_matrix(cm, name, plot_dir)

    if len(results_dict) > 1:
        plot_all_confusion_matrices(cms_dict, plot_dir)
        plot_metrics_comparison(results_dict, plot_dir)

    # robustness testing on best available config
    if results_dict:
        best_config = max(results_dict, key=lambda k: results_dict[k]["f1"])
        print(f"\nrunning robustness tests on best config: {best_config}")
        best_model = load_model(
            config,
            os.path.join(ckpt_dir, best_config, "best.pt")
        ).to(device)

        for perturbation in ["gaussian_noise", "blur", "occlusion"]:
            m = run_perturbation_test(best_model, test_dl, device, perturbation)
            print(f"  {perturbation}: f1={m['f1']:.4f} iou={m['iou']:.4f}")

        # save robustness results
        rob_path = os.path.join(plot_dir, "robustness_results.txt")
        with open(rob_path, "w") as f:
            f.write(f"best config: {best_config}\n")
            for perturbation in ["gaussian_noise", "blur", "occlusion"]:
                m = run_perturbation_test(best_model, test_dl, device, perturbation)
                f.write(f"{perturbation}: f1={m['f1']:.4f} iou={m['iou']:.4f}\n")
        print(f"robustness results saved to {rob_path}")

    print("\nevaluation complete!")

if __name__ == "__main__":
    main()
