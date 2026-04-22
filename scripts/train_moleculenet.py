import csv
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader as TorchDataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graph_set_transformer.data import (
    BalancedSetBatchSampler,
    SetDataset,
    collate_sets,
    make_label_homogeneous_sets,
    make_label_homogeneous_sets_rand_card,
)
from graph_set_transformer.models import (
    DeepSetGraphMultiTask,
    SetGraphMultiTask,
    SetTransformerGraphMultiTask,
)
from graph_set_transformer.utils import molecule_net_loader


SET_SIZE_CONFIGS = [
    {"name": "rand_1_5", "mode": "random_range", "min_size": 1, "max_size": 5},
    {"name": "rand_1_10", "mode": "random_range", "min_size": 1, "max_size": 10},
    {"name": "fixed_5", "mode": "fixed", "size": 5},
    {"name": "fixed_10", "mode": "fixed", "size": 10},
    {"name": "fixed_20", "mode": "fixed", "size": 20},
]


def get_model(model_name, in_channels, hidden_dim):
    if model_name == "SetTransformer":
        return SetTransformerGraphMultiTask(in_channels, hidden_dim)
    if model_name == "DeepSets":
        return DeepSetGraphMultiTask(in_channels, hidden_dim)
    if model_name == "GraphSetConv":
        return SetGraphMultiTask(in_channels, hidden_dim)
    raise ValueError(f"Unknown model: {model_name}")


def build_sets(dataset, set_config, shuffle=False):
    if set_config["mode"] == "fixed":
        return make_label_homogeneous_sets(dataset, set_config["size"], shuffle=shuffle)

    return make_label_homogeneous_sets_rand_card(
        dataset,
        min_size=set_config["min_size"],
        max_size=set_config["max_size"],
        shuffle=shuffle,
    )


def describe_set_config(set_config):
    if set_config["mode"] == "fixed":
        return str(set_config["size"])
    return f"{set_config['min_size']}-{set_config['max_size']} random"


def append_prediction_rows(csv_path, rows):
    if not rows:
        return

    fieldnames = [
        "epoch",
        "phase",
        "prediction_type",
        "sample_index",
        "set_index",
        "target_value",
        "prediction_value",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def build_prediction_rows(
    phase,
    epoch,
    set_labels,
    set_logits,
    graph_regression_targets,
    graph_regression,
    set_batch,
):
    rows = []
    set_probs = torch.sigmoid(set_logits).detach().cpu().numpy()
    set_targets = set_labels.detach().cpu().numpy()
    graph_targets = graph_regression_targets.detach().cpu().numpy()
    graph_preds = graph_regression.detach().cpu().numpy()
    graph_set_indices = set_batch.detach().cpu().numpy()

    for sample_index, (target_value, prediction_value) in enumerate(
        zip(set_targets, set_probs)
    ):
        rows.append(
            {
                "epoch": epoch,
                "phase": phase,
                "prediction_type": "set_classification",
                "sample_index": sample_index,
                "set_index": sample_index,
                "target_value": float(target_value),
                "prediction_value": float(prediction_value),
            }
        )

    for sample_index, (target_value, prediction_value, set_index) in enumerate(
        zip(graph_targets, graph_preds, graph_set_indices)
    ):
        rows.append(
            {
                "epoch": epoch,
                "phase": phase,
                "prediction_type": "graph_regression",
                "sample_index": sample_index,
                "set_index": int(set_index),
                "target_value": float(target_value),
                "prediction_value": float(prediction_value),
            }
        )

    return rows


def train_epoch(model, loader, optimizer, device, bce_loss_fn, mse_loss_fn, epoch=None):
    model.train()
    totals = {"total_loss": 0.0, "bce_loss": 0.0, "mse_loss": 0.0, "num_batches": 0}
    prediction_rows = []

    for data, set_batch, set_labels, graph_regression_targets in loader:
        data = data.to(device)
        set_batch = set_batch.to(device)
        set_labels = set_labels.float().to(device)
        graph_regression_targets = graph_regression_targets.to(device)

        optimizer.zero_grad()
        set_logits, graph_regression = model(data, set_batch)

        bce_loss = bce_loss_fn(set_logits, set_labels)
        mse_loss = mse_loss_fn(graph_regression, graph_regression_targets)
        total_loss = bce_loss + mse_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch is not None:
            prediction_rows.extend(
                build_prediction_rows(
                    phase="train",
                    epoch=epoch,
                    set_labels=set_labels,
                    set_logits=set_logits,
                    graph_regression_targets=graph_regression_targets,
                    graph_regression=graph_regression,
                    set_batch=set_batch,
                )
            )

        totals["total_loss"] += total_loss.item()
        totals["bce_loss"] += bce_loss.item()
        totals["mse_loss"] += mse_loss.item()
        totals["num_batches"] += 1

    for key in ("total_loss", "bce_loss", "mse_loss"):
        totals[key] /= max(totals["num_batches"], 1)

    return totals, prediction_rows


def evaluate(model, loader, device, bce_loss_fn, mse_loss_fn, phase="valid", epoch=None):
    model.eval()
    totals = {"total_loss": 0.0, "bce_loss": 0.0, "mse_loss": 0.0, "num_batches": 0}
    all_probs = []
    all_set_labels = []
    prediction_rows = []

    with torch.no_grad():
        for data, set_batch, set_labels, graph_regression_targets in loader:
            data = data.to(device)
            set_batch = set_batch.to(device)
            set_labels = set_labels.float().to(device)
            graph_regression_targets = graph_regression_targets.to(device)

            set_logits, graph_regression = model(data, set_batch)
            bce_loss = bce_loss_fn(set_logits, set_labels)
            mse_loss = mse_loss_fn(graph_regression, graph_regression_targets)
            total_loss = bce_loss + mse_loss

            probs = torch.sigmoid(set_logits)
            all_probs.extend(probs.cpu().numpy())
            all_set_labels.extend(set_labels.cpu().numpy())

            if epoch is not None:
                prediction_rows.extend(
                    build_prediction_rows(
                        phase=phase,
                        epoch=epoch,
                        set_labels=set_labels,
                        set_logits=set_logits,
                        graph_regression_targets=graph_regression_targets,
                        graph_regression=graph_regression,
                        set_batch=set_batch,
                    )
                )

            totals["total_loss"] += total_loss.item()
            totals["bce_loss"] += bce_loss.item()
            totals["mse_loss"] += mse_loss.item()
            totals["num_batches"] += 1

    for key in ("total_loss", "bce_loss", "mse_loss"):
        totals[key] /= max(totals["num_batches"], 1)

    unique_labels = np.unique(all_set_labels)
    totals["auroc"] = (
        roc_auc_score(all_set_labels, all_probs) if unique_labels.size > 1 else float("nan")
    )
    return totals, prediction_rows


def clone_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def main():
    seeds = [10, 20, 30, 40, 50]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_names = ["SetTransformer", "DeepSets", "GraphSetConv"]
    num_epochs = 500
    hidden_dim = 64
    batch_size = 32
    learning_rate = 1e-4
    results_root = Path("results") / "BACE_DoubleTask_BalanceData_1"
    results_root.mkdir(parents=True, exist_ok=True)

    bce_loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()

    all_results = {
        set_config["name"]: {
            model_name: {
                "train_total_loss_per_seed": [],
                "train_bce_per_seed": [],
                "train_mse_per_seed": [],
                "val_total_loss_per_seed": [],
                "val_auroc_per_seed": [],
                "val_mse_per_seed": [],
                "test_total_loss_per_seed": [],
                "test_auroc_per_seed": [],
                "test_mse_per_seed": [],
            }
            for model_name in model_names
        }
        for set_config in SET_SIZE_CONFIGS
    }

    train_dataset, val_dataset, test_dataset, tasks = molecule_net_loader(
        "bace",
        "data/moleculenet/bace_split_8_1_1_seed42.csv.xz",
        split_mode="official",
        y_columns=["Class", "pIC50"],
        label_dtype=torch.float32,
    )
    print(f"Loaded multitask targets: {tasks}")

    in_channels = train_dataset[0].x.shape[1]

    for set_config in SET_SIZE_CONFIGS:
        set_name = set_config["name"]
        set_desc = describe_set_config(set_config)
        set_results_dir = results_root / set_name
        set_results_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'*' * 70}")
        print(f"* SET CONFIG: {set_name} ({set_desc})")
        print(f"{'*' * 70}")

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            seed_results_dir = set_results_dir / f"seed_{seed}"
            seed_results_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'#' * 60}")
            print(f"# SET CONFIG: {set_name} - SEED {seed}")
            print(f"{'#' * 60}")

            val_sets = build_sets(val_dataset, set_config, shuffle=False)
            test_sets = build_sets(test_dataset, set_config, shuffle=False)
            val_loader = TorchDataLoader(
                SetDataset(val_sets),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_sets,
            )
            test_loader = TorchDataLoader(
                SetDataset(test_sets),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_sets,
            )
            print(
                f"Validation/Test sets for {set_name}: "
                f"Val={len(val_sets)}, Test={len(test_sets)}"
            )

            trial_results = {
                model_name: {
                    "train_total_loss": [],
                    "train_bce": [],
                    "train_mse": [],
                    "val_total_loss": [],
                    "val_auroc": [],
                    "val_mse": [],
                    "test_total_loss": [],
                    "test_auroc": [],
                    "test_mse": [],
                }
                for model_name in model_names
            }

            for model_name in model_names:
                print(f"\n{'=' * 50}")
                print(f"{set_name} - Seed {seed} - Training {model_name}")
                print(f"{'=' * 50}")

                model = get_model(model_name, in_channels, hidden_dim).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                model_results_dir = seed_results_dir / model_name
                model_results_dir.mkdir(parents=True, exist_ok=True)
                prediction_csv_path = model_results_dir / "train_valid_test_predictions.csv"
                epoch_metrics_csv_path = model_results_dir / "epoch_metrics.csv"
                best_test_results_csv_path = model_results_dir / "test_results.csv"
                if prediction_csv_path.exists():
                    prediction_csv_path.unlink()
                if epoch_metrics_csv_path.exists():
                    epoch_metrics_csv_path.unlink()
                if best_test_results_csv_path.exists():
                    best_test_results_csv_path.unlink()

                best_val_total_loss = float("inf")
                best_model_state = None

                for epoch in range(num_epochs):
                    train_sets = build_sets(train_dataset, set_config, shuffle=True)
                    train_set_dataset = SetDataset(train_sets)

                    balanced_sampler = BalancedSetBatchSampler(
                        train_set_dataset,
                        batch_size=batch_size,
                        num_classes=2,
                    )

                    train_loader = TorchDataLoader(
                        train_set_dataset,
                        batch_sampler=balanced_sampler,
                        collate_fn=collate_sets,
                    )

                    if epoch == 0 and seed == seeds[0]:
                        print("\nVerifying first training batch:")
                        first_batch = next(iter(train_loader))
                        _, _, set_labels, graph_regression_targets = first_batch
                        print(
                            "  set_labels.shape=",
                            tuple(set_labels.shape),
                            "graph_regression_targets.shape=",
                            tuple(graph_regression_targets.shape),
                        )
                        print(
                            "  batch class counts=",
                            torch.bincount(set_labels, minlength=2).tolist(),
                        )

                    train_metrics, train_prediction_rows = train_epoch(
                        model,
                        train_loader,
                        optimizer,
                        device,
                        bce_loss_fn,
                        mse_loss_fn,
                        epoch=epoch + 1,
                    )
                    val_metrics, val_prediction_rows = evaluate(
                        model,
                        val_loader,
                        device,
                        bce_loss_fn,
                        mse_loss_fn,
                        phase="valid",
                        epoch=epoch + 1,
                    )
                    test_metrics, test_prediction_rows = evaluate(
                        model,
                        test_loader,
                        device,
                        bce_loss_fn,
                        mse_loss_fn,
                        phase="test",
                        epoch=epoch + 1,
                    )
                    append_prediction_rows(
                        prediction_csv_path,
                        train_prediction_rows + val_prediction_rows + test_prediction_rows,
                    )
                    pd.DataFrame(
                        [
                            {
                                "epoch": epoch + 1,
                                "train_total_loss": train_metrics["total_loss"],
                                "train_bce": train_metrics["bce_loss"],
                                "train_mse": train_metrics["mse_loss"],
                                "valid_total_loss": val_metrics["total_loss"],
                                "valid_auroc": val_metrics["auroc"],
                                "valid_mse": val_metrics["mse_loss"],
                                "test_total_loss": test_metrics["total_loss"],
                                "test_auroc": test_metrics["auroc"],
                                "test_mse": test_metrics["mse_loss"],
                            }
                        ]
                    ).to_csv(
                        epoch_metrics_csv_path,
                        mode="a",
                        index=False,
                        header=not epoch_metrics_csv_path.exists(),
                    )

                    trial_results[model_name]["train_total_loss"].append(
                        train_metrics["total_loss"]
                    )
                    trial_results[model_name]["train_bce"].append(train_metrics["bce_loss"])
                    trial_results[model_name]["train_mse"].append(train_metrics["mse_loss"])
                    trial_results[model_name]["val_total_loss"].append(
                        val_metrics["total_loss"]
                    )
                    trial_results[model_name]["val_auroc"].append(val_metrics["auroc"])
                    trial_results[model_name]["val_mse"].append(val_metrics["mse_loss"])
                    trial_results[model_name]["test_total_loss"].append(
                        test_metrics["total_loss"]
                    )
                    trial_results[model_name]["test_auroc"].append(test_metrics["auroc"])
                    trial_results[model_name]["test_mse"].append(test_metrics["mse_loss"])

                    if val_metrics["total_loss"] < best_val_total_loss:
                        best_val_total_loss = val_metrics["total_loss"]
                        best_model_state = clone_state_dict(model)

                    if (epoch + 1) % 10 == 0:
                        print(
                            f"Epoch {epoch + 1}/{num_epochs} - "
                            f"Train total: {train_metrics['total_loss']:.4f}, "
                            f"Train BCE: {train_metrics['bce_loss']:.4f}, "
                            f"Train MSE: {train_metrics['mse_loss']:.4f}, "
                            f"Val total: {val_metrics['total_loss']:.4f}, "
                            f"Val AUROC: {val_metrics['auroc']:.4f}, "
                            f"Val MSE: {val_metrics['mse_loss']:.4f}, "
                            f"Test total: {test_metrics['total_loss']:.4f}, "
                            f"Test AUROC: {test_metrics['auroc']:.4f}, "
                            f"Test MSE: {test_metrics['mse_loss']:.4f}"
                        )

                print(f"Best Val Total Loss for {model_name}: {best_val_total_loss:.4f}")

                model.load_state_dict(best_model_state)
                torch.save(best_model_state, model_results_dir / "best_model.pt")

                test_metrics, best_test_prediction_rows = evaluate(
                    model,
                    test_loader,
                    device,
                    bce_loss_fn,
                    mse_loss_fn,
                    phase="best_test",
                    epoch="best",
                )
                pd.DataFrame(best_test_prediction_rows).to_csv(
                    best_test_results_csv_path, index=False
                )
                print(
                    f"Test total: {test_metrics['total_loss']:.4f}, "
                    f"Test AUROC: {test_metrics['auroc']:.4f}, "
                    f"Test MSE: {test_metrics['mse_loss']:.4f}"
                )
                pd.DataFrame(
                    {
                        "seed": [seed],
                        "set_config": [set_name],
                        "set_description": [set_desc],
                        "model": [model_name],
                        "best_val_total_loss": [best_val_total_loss],
                        "test_total_loss": [test_metrics["total_loss"]],
                        "test_auroc": [test_metrics["auroc"]],
                        "test_mse": [test_metrics["mse_loss"]],
                    }
                ).to_csv(model_results_dir / "metrics_summary.csv", index=False)

                all_results[set_name][model_name]["train_total_loss_per_seed"].append(
                    trial_results[model_name]["train_total_loss"]
                )
                all_results[set_name][model_name]["train_bce_per_seed"].append(
                    trial_results[model_name]["train_bce"]
                )
                all_results[set_name][model_name]["train_mse_per_seed"].append(
                    trial_results[model_name]["train_mse"]
                )
                all_results[set_name][model_name]["val_total_loss_per_seed"].append(
                    trial_results[model_name]["val_total_loss"]
                )
                all_results[set_name][model_name]["val_auroc_per_seed"].append(
                    trial_results[model_name]["val_auroc"]
                )
                all_results[set_name][model_name]["val_mse_per_seed"].append(
                    trial_results[model_name]["val_mse"]
                )
                all_results[set_name][model_name]["test_total_loss_per_seed"].append(
                    test_metrics["total_loss"]
                )
                all_results[set_name][model_name]["test_auroc_per_seed"].append(
                    test_metrics["auroc"]
                )
                all_results[set_name][model_name]["test_mse_per_seed"].append(
                    test_metrics["mse_loss"]
                )

    print(f"\n{'=' * 70}")
    print("FINAL RESULTS ACROSS ALL SET CONFIGS, MODELS, AND TRIALS")
    print(f"{'=' * 70}\n")

    summary_data = []
    for set_config in SET_SIZE_CONFIGS:
        set_name = set_config["name"]
        set_desc = describe_set_config(set_config)
        print(f"\nSET CONFIG: {set_name} ({set_desc})")
        print(f"{'-' * 60}")

        for model_name in model_names:
            test_aurocs = all_results[set_name][model_name]["test_auroc_per_seed"]
            test_mses = all_results[set_name][model_name]["test_mse_per_seed"]
            test_total_losses = all_results[set_name][model_name]["test_total_loss_per_seed"]

            print(f"{model_name}:")
            print(f"  Test AUROC: {test_aurocs}")
            print(f"  Test MSE: {test_mses}")
            print(f"  Test total loss: {test_total_losses}")
            print(
                f"  Mean AUROC ± Std: {np.nanmean(test_aurocs):.4f} ± {np.nanstd(test_aurocs):.4f}"
            )
            print(
                f"  Mean MSE ± Std: {np.mean(test_mses):.4f} ± {np.std(test_mses):.4f}"
            )
            print(
                f"  Mean total loss ± Std: "
                f"{np.mean(test_total_losses):.4f} ± {np.std(test_total_losses):.4f}\n"
            )

            summary_data.append(
                {
                    "Set Config": set_name,
                    "Set Description": set_desc,
                    "Model": model_name,
                    "Mean Test Total Loss": np.mean(test_total_losses),
                    "Std Test Total Loss": np.std(test_total_losses),
                    "Mean Test AUROC": np.nanmean(test_aurocs),
                    "Std Test AUROC": np.nanstd(test_aurocs),
                    "Mean Test MSE": np.mean(test_mses),
                    "Std Test MSE": np.std(test_mses),
                    "All Test Total Losses": test_total_losses,
                    "All Test AUROCs": test_aurocs,
                    "All Test MSEs": test_mses,
                }
            )

    for set_config in SET_SIZE_CONFIGS:
        set_name = set_config["name"]
        fig, axes = plt.subplots(len(seeds), 3, figsize=(18, 5 * len(seeds)))

        for row_idx, seed in enumerate(seeds):
            for model_name in model_names:
                axes[row_idx, 0].plot(
                    all_results[set_name][model_name]["train_total_loss_per_seed"][row_idx],
                    label=model_name,
                )
                axes[row_idx, 1].plot(
                    all_results[set_name][model_name]["val_auroc_per_seed"][row_idx],
                    label=model_name,
                )
                axes[row_idx, 2].plot(
                    all_results[set_name][model_name]["val_mse_per_seed"][row_idx],
                    label=model_name,
                )

            axes[row_idx, 0].set_title(f"Train Total Loss ({set_name}, Seed {seed})")
            axes[row_idx, 0].set_xlabel("Epoch")
            axes[row_idx, 0].set_ylabel("Loss")
            axes[row_idx, 0].legend()

            axes[row_idx, 1].set_title(f"Validation AUROC ({set_name}, Seed {seed})")
            axes[row_idx, 1].set_xlabel("Epoch")
            axes[row_idx, 1].set_ylabel("AUROC")
            axes[row_idx, 1].legend()

            axes[row_idx, 2].set_title(f"Validation MSE ({set_name}, Seed {seed})")
            axes[row_idx, 2].set_xlabel("Epoch")
            axes[row_idx, 2].set_ylabel("MSE")
            axes[row_idx, 2].legend()

        plt.tight_layout()
        plot_filename = results_root / f"model_comparison_{set_name}.png"
        plt.savefig(plot_filename)
        plt.close(fig)
        print(f"Saved plot to {plot_filename}")

    summary_df = pd.DataFrame(summary_data)
    csv_filename = results_root / "model_comparison_multitask_set_sizes.csv"
    summary_df.to_csv(csv_filename, index=False)
    print(f"Saved summary to {csv_filename}")
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(
        summary_df[
            [
                "Set Config",
                "Model",
                "Mean Test Total Loss",
                "Mean Test AUROC",
                "Mean Test MSE",
            ]
        ]
    )
    print("\n")


if __name__ == "__main__":
    main()
