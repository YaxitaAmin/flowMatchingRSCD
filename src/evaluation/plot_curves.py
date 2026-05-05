import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def plot_loss_curve(log_path, config_name, save_dir):
    if not os.path.exists(log_path):
        print(f"log not found: {log_path}")
        return

    data = np.loadtxt(log_path, delimiter=",", skiprows=1)
    if data.size == 0:
        print(f"skipping {config_name}")
        return
    if data.ndim == 1:
        data = data[np.newaxis, :]

    epochs     = data[:, 0]
    train_loss = data[:, 1]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_loss, label="train loss", color="steelblue")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"training loss: {config_name}")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(save_dir, f"loss_{config_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved: {path}")

def plot_cd_curves(log_path, config_name, save_dir):
    if not os.path.exists(log_path):
        print(f"log not found: {log_path}")
        return

    data = np.loadtxt(log_path, delimiter=",", skiprows=1)
    if data.size == 0:
        print(f"skipping {config_name}")
        return
    if data.ndim == 1:
        data = data[np.newaxis, :]

    epochs     = data[:, 0]
    train_loss = data[:, 1]
    val_f1     = data[:, 2]
    val_iou    = data[:, 3]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, color="steelblue", label="train loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"train loss: {config_name}")
    axes[0].legend()

    axes[1].plot(epochs, val_f1,  color="darkorange", label="val f1")
    axes[1].plot(epochs, val_iou, color="green",      label="val iou")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].set_title(f"val metrics: {config_name}")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(save_dir, f"curves_{config_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved: {path}")

def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    log_dir  = config["paths"]["logs"]
    plot_dir = config["paths"]["plots"]
    os.makedirs(plot_dir, exist_ok=True)

    # flow matching loss curve
    plot_loss_curve(
        os.path.join(log_dir, "flow_matching_train.log"),
        "flow_matching",
        plot_dir
    )

    # change detection curves for all configs
    for name in ["config1_baseline", "config2_augmented",
                 "config3_flow_matching", "config4_full_pipeline"]:
        plot_cd_curves(
            os.path.join(log_dir, f"cd_{name}.log"),
            name,
            plot_dir
        )

    print("all plots saved!")

if __name__ == "__main__":
    main()
