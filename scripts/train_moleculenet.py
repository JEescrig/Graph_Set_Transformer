import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader as TorchDataLoader
from rdkit import RDLogger
from rdkit.Chem.AllChem import MolFromSmiles

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graph_set_transformer.data import (  # noqa: E402
    BalancedSetBatchSampler,
    SetDataset,
    collate_sets,
    make_label_homogeneous_sets,
    make_label_homogeneous_sets_rand_card,
)
from graph_set_transformer.models import (  # noqa: E402
    DeepSetGraphSetElementClassifier,
    SetGraphSetElementClassifier,
    SetTransformerGraphSetElementClassifier,
)
from graph_set_transformer.utils.graph_encoder import GraphEncoder  # noqa: E402


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    display_name: str
    path: Path
    smiles_column: str
    label_column: str
    set_config_names: tuple[str, ...]


@dataclass(frozen=True)
class SetConfig:
    name: str
    table_label: str
    mode: str
    size: int | None = None
    min_size: int | None = None
    max_size: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    display_name: str
    factory: object


SEEDS = [10, 20, 30, 40, 50]

SET_CONFIGS = [
    SetConfig("fixed_20", "20", "fixed", size=20),
    SetConfig("fixed_10", "10", "fixed", size=10),
    SetConfig("fixed_5", "5", "fixed", size=5),
    SetConfig("rand_1_10", "1-10", "random_range", min_size=1, max_size=10),
    SetConfig("rand_1_5", "1-5", "random_range", min_size=1, max_size=5),
]
SET_CONFIG_BY_NAME = {config.name: config for config in SET_CONFIGS}

DATASETS = [
    DatasetConfig(
        dataset_id="BACE",
        display_name="BACE",
        path=Path("data/moleculenet/bace.csv.xz"),
        smiles_column="smiles",
        label_column="Class",
        set_config_names=("fixed_20", "fixed_10", "fixed_5", "rand_1_10", "rand_1_5"),
    ),
    DatasetConfig(
        dataset_id="BBBP",
        display_name="BBBP",
        path=Path("data/moleculenet/bbbp.csv.xz"),
        smiles_column="smiles",
        label_column="p_np",
        set_config_names=("fixed_20", "fixed_10", "fixed_5", "rand_1_10", "rand_1_5"),
    ),
    DatasetConfig(
        dataset_id="Pgp_Broccatelli",
        display_name="Pgp Broccatelli",
        path=Path("data/Pgp_Broccatelli/Pgp_Broccatelli.csv"),
        smiles_column="Drug",
        label_column="Y",
        set_config_names=("fixed_20", "fixed_10", "fixed_5", "rand_1_10", "rand_1_5"),
    ),
    DatasetConfig(
        dataset_id="BBB_Martins",
        display_name="BBB Martins",
        path=Path("data/BBB_Martins/BBB_Martins.csv"),
        smiles_column="Drug",
        label_column="Y",
        set_config_names=("fixed_20", "fixed_10", "fixed_5", "rand_1_10", "rand_1_5"),
    ),
    DatasetConfig(
        dataset_id="CYP3A4",
        display_name="CYP3A4",
        path=Path("data/CYP3A4_Substrate/CYP3A4_Substrate_CarbonMangels.csv"),
        smiles_column="Drug",
        label_column="Y",
        set_config_names=("fixed_5", "rand_1_5"),
    ),
]
DATASET_BY_ID = {config.dataset_id: config for config in DATASETS}

MODEL_CONFIGS = [
    ModelConfig(
        "GCN_DeepSets",
        "GCN + Deep Sets",
        lambda in_channels, hidden_dim: DeepSetGraphSetElementClassifier(
            in_channels, hidden_dim
        ),
    ),
    ModelConfig(
        "GCN_SetTransformer",
        "GCN + SetTransformer",
        lambda in_channels, hidden_dim: SetTransformerGraphSetElementClassifier(
            in_channels, hidden_dim
        ),
    ),
    ModelConfig(
        "GST",
        "GST (ours)",
        lambda in_channels, hidden_dim: SetGraphSetElementClassifier(
            in_channels, hidden_dim
        ),
    ),
]
MODEL_BY_ID = {config.model_id: config for config in MODEL_CONFIGS}


def is_supported_smiles(smiles):
    smiles = str(smiles)
    mol = MolFromSmiles(smiles)
    if mol is None:
        mol = MolFromSmiles(smiles.replace("[NH+2]", "[NH+1]"))
    return mol is not None and mol.GetNumBonds() > 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train set-level and element-level classifiers on five datasets."
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_BY_ID),
        default=list(DATASET_BY_ID),
    )
    parser.add_argument(
        "--set-configs",
        nargs="+",
        choices=list(SET_CONFIG_BY_NAME),
        default=None,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_BY_ID),
        default=list(MODEL_BY_ID),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results") / "MoleculeNet_SetElement_Classification_8_1_1",
    )
    parser.add_argument(
        "--skip-epoch-predictions",
        action="store_true",
        help="Skip train/valid/test prediction CSVs during every epoch.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_frame(config):
    RDLogger.DisableLog("rdApp.*")

    if not config.path.exists():
        raise FileNotFoundError(f"Missing dataset file: {config.path}")

    df = pd.read_csv(config.path)
    required_columns = [config.smiles_column, config.label_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{config.dataset_id} is missing required columns: {missing_columns}"
        )

    df = df[[config.smiles_column, config.label_column]].copy()
    df.insert(0, "original_index", df.index)
    df = df.rename(
        columns={
            config.smiles_column: "smiles",
            config.label_column: "label",
        }
    )
    df = df.dropna(subset=["smiles", "label"]).reset_index(drop=True)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    df = df[df["smiles"].map(is_supported_smiles)].reset_index(drop=True)

    if df["label"].nunique() != 2:
        raise ValueError(f"{config.dataset_id} does not contain both binary classes.")

    return df


def split_frame_8_1_1(df, seed):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(df))
    train_end = int(0.8 * len(df))
    valid_end = train_end + int(0.1 * len(df))

    split_frames = {
        "train": df.iloc[shuffled[:train_end]].reset_index(drop=True),
        "valid": df.iloc[shuffled[train_end:valid_end]].reset_index(drop=True),
        "test": df.iloc[shuffled[valid_end:]].reset_index(drop=True),
    }
    return split_frames


def save_split_indices(split_frames, path):
    rows = []
    for split_name, split_df in split_frames.items():
        for row_position, original_index in enumerate(split_df["original_index"]):
            rows.append(
                {
                    "split": split_name,
                    "row_position": row_position,
                    "original_index": int(original_index),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def encode_split_frames(split_frames):
    encoder = GraphEncoder()
    encoded = {}
    for split_name, split_df in split_frames.items():
        encoded[split_name] = encoder.encode(
            split_df["smiles"].to_numpy(),
            split_df["label"].to_numpy(dtype=np.float32),
            label_dtype=torch.float32,
        )
        if not encoded[split_name]:
            raise ValueError(f"No valid molecules encoded for {split_name} split.")
    return encoded


def build_sets(dataset, set_config, shuffle=False):
    if set_config.mode == "fixed":
        return make_label_homogeneous_sets(dataset, set_config.size, shuffle=shuffle)

    return make_label_homogeneous_sets_rand_card(
        dataset,
        min_size=set_config.min_size,
        max_size=set_config.max_size,
        shuffle=shuffle,
    )


def make_loader(sets, batch_size, shuffle=False, balanced=False):
    set_dataset = SetDataset(sets)
    if balanced:
        return TorchDataLoader(
            set_dataset,
            batch_sampler=BalancedSetBatchSampler(
                set_dataset,
                batch_size=batch_size,
                num_classes=2,
            ),
            collate_fn=collate_sets,
        )

    return TorchDataLoader(
        set_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_sets,
    )


def safe_roc_auc(targets, predictions):
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    if targets.size == 0 or np.unique(targets).size < 2:
        return float("nan")
    return roc_auc_score(targets, predictions)


def append_prediction_rows(csv_path, rows):
    if not rows:
        return

    fieldnames = [
        "epoch",
        "phase",
        "batch_index",
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
    batch_index,
    set_labels,
    set_logits,
    element_labels,
    element_logits,
    set_batch,
):
    rows = []
    set_probs = torch.sigmoid(set_logits).detach().cpu().numpy()
    set_targets = set_labels.detach().cpu().numpy()
    element_probs = torch.sigmoid(element_logits).detach().cpu().numpy()
    element_targets = element_labels.detach().cpu().numpy()
    element_set_indices = set_batch.detach().cpu().numpy()

    for sample_index, (target_value, prediction_value) in enumerate(
        zip(set_targets, set_probs)
    ):
        rows.append(
            {
                "epoch": epoch,
                "phase": phase,
                "batch_index": batch_index,
                "prediction_type": "set_classification",
                "sample_index": sample_index,
                "set_index": sample_index,
                "target_value": float(target_value),
                "prediction_value": float(prediction_value),
            }
        )

    for sample_index, (target_value, prediction_value, set_index) in enumerate(
        zip(element_targets, element_probs, element_set_indices)
    ):
        rows.append(
            {
                "epoch": epoch,
                "phase": phase,
                "batch_index": batch_index,
                "prediction_type": "element_classification",
                "sample_index": sample_index,
                "set_index": int(set_index),
                "target_value": float(target_value),
                "prediction_value": float(prediction_value),
            }
        )

    return rows


def compute_losses(
    set_logits,
    element_logits,
    set_labels,
    element_labels,
    bce_loss_fn,
):
    set_targets = set_labels.float()
    element_targets = element_labels.float()
    set_bce = bce_loss_fn(set_logits, set_targets)
    element_bce = bce_loss_fn(element_logits, element_targets)
    total_loss = set_bce + element_bce
    return total_loss, set_bce, element_bce


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    bce_loss_fn,
    epoch=None,
    collect_predictions=True,
):
    model.train()
    totals = {
        "total_loss": 0.0,
        "set_bce": 0.0,
        "element_bce": 0.0,
        "num_batches": 0,
    }
    prediction_rows = []

    for batch_index, (data, set_batch, set_labels, element_labels) in enumerate(loader):
        data = data.to(device)
        set_batch = set_batch.to(device)
        set_labels = set_labels.to(device)
        element_labels = element_labels.to(device)

        optimizer.zero_grad()
        set_logits, element_logits = model(data, set_batch)
        total_loss, set_bce, element_bce = compute_losses(
            set_logits,
            element_logits,
            set_labels,
            element_labels,
            bce_loss_fn,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if collect_predictions and epoch is not None:
            prediction_rows.extend(
                build_prediction_rows(
                    phase="train",
                    epoch=epoch,
                    batch_index=batch_index,
                    set_labels=set_labels,
                    set_logits=set_logits,
                    element_labels=element_labels,
                    element_logits=element_logits,
                    set_batch=set_batch,
                )
            )

        totals["total_loss"] += total_loss.item()
        totals["set_bce"] += set_bce.item()
        totals["element_bce"] += element_bce.item()
        totals["num_batches"] += 1

    for key in ("total_loss", "set_bce", "element_bce"):
        totals[key] /= max(totals["num_batches"], 1)

    return totals, prediction_rows


def evaluate(
    model,
    loader,
    device,
    bce_loss_fn,
    phase="valid",
    epoch=None,
    collect_predictions=True,
):
    model.eval()
    totals = {
        "total_loss": 0.0,
        "set_bce": 0.0,
        "element_bce": 0.0,
        "num_batches": 0,
    }
    all_set_probs = []
    all_set_labels = []
    all_element_probs = []
    all_element_labels = []
    prediction_rows = []

    with torch.no_grad():
        for batch_index, (data, set_batch, set_labels, element_labels) in enumerate(
            loader
        ):
            data = data.to(device)
            set_batch = set_batch.to(device)
            set_labels = set_labels.to(device)
            element_labels = element_labels.to(device)

            set_logits, element_logits = model(data, set_batch)
            total_loss, set_bce, element_bce = compute_losses(
                set_logits,
                element_logits,
                set_labels,
                element_labels,
                bce_loss_fn,
            )

            set_probs = torch.sigmoid(set_logits)
            element_probs = torch.sigmoid(element_logits)
            all_set_probs.extend(set_probs.cpu().numpy())
            all_set_labels.extend(set_labels.cpu().numpy())
            all_element_probs.extend(element_probs.cpu().numpy())
            all_element_labels.extend(element_labels.cpu().numpy())

            if collect_predictions and epoch is not None:
                prediction_rows.extend(
                    build_prediction_rows(
                        phase=phase,
                        epoch=epoch,
                        batch_index=batch_index,
                        set_labels=set_labels,
                        set_logits=set_logits,
                        element_labels=element_labels,
                        element_logits=element_logits,
                        set_batch=set_batch,
                    )
                )

            totals["total_loss"] += total_loss.item()
            totals["set_bce"] += set_bce.item()
            totals["element_bce"] += element_bce.item()
            totals["num_batches"] += 1

    for key in ("total_loss", "set_bce", "element_bce"):
        totals[key] /= max(totals["num_batches"], 1)

    totals["set_auroc"] = safe_roc_auc(all_set_labels, all_set_probs)
    totals["element_auroc"] = safe_roc_auc(all_element_labels, all_element_probs)
    return totals, prediction_rows


def clone_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def write_epoch_metrics(path, row):
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        index=False,
        header=not path.exists(),
    )


def format_metric(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.all(np.isnan(values)):
        return "-"
    return f"{np.nanmean(values) * 100:.1f} +/- {np.nanstd(values) * 100:.1f}"


def write_markdown_table(df, path):
    headers = list(df.columns)
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    path.write_text("\n".join(rows) + "\n")


def build_summary_tables(summary_rows, results_root):
    raw_df = pd.DataFrame(summary_rows)
    raw_path = results_root / "summary_raw.csv"
    raw_df.to_csv(raw_path, index=False)

    table_rows = []
    for dataset_config in DATASETS:
        for set_config in SET_CONFIGS:
            table_row = {
                "Data Set": dataset_config.display_name,
                "|S|": set_config.table_label,
            }
            subset = raw_df[
                (raw_df["dataset"] == dataset_config.dataset_id)
                & (raw_df["set_config"] == set_config.name)
            ]
            for model_config in MODEL_CONFIGS:
                model_subset = subset[subset["model"] == model_config.model_id]
                table_row[
                    f"{model_config.display_name} Set ROC-AUC"
                ] = format_metric(model_subset["test_set_auroc"].to_numpy())
                table_row[
                    f"{model_config.display_name} Element ROC-AUC"
                ] = format_metric(model_subset["test_element_auroc"].to_numpy())
            table_rows.append(table_row)

    table_df = pd.DataFrame(table_rows)
    table_csv_path = results_root / "summary_table.csv"
    table_md_path = results_root / "summary_table.md"
    table_df.to_csv(table_csv_path, index=False)
    write_markdown_table(table_df, table_md_path)
    return raw_path, table_csv_path, table_md_path


def selected_set_configs(dataset_config, requested_set_configs):
    allowed = [SET_CONFIG_BY_NAME[name] for name in dataset_config.set_config_names]
    if requested_set_configs is None:
        return allowed
    requested = set(requested_set_configs)
    return [config for config in allowed if config.name in requested]


def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    args.results_root.mkdir(parents=True, exist_ok=True)

    selected_datasets = [DATASET_BY_ID[dataset_id] for dataset_id in args.datasets]
    selected_models = [MODEL_BY_ID[model_id] for model_id in args.models]
    bce_loss_fn = nn.BCEWithLogitsLoss()
    summary_rows = []

    for dataset_config in selected_datasets:
        df = load_dataset_frame(dataset_config)
        print(
            f"\nDataset {dataset_config.display_name}: rows={len(df)}, "
            f"class_counts={df['label'].value_counts().sort_index().to_dict()}"
        )

        for seed in args.seeds:
            set_seed(seed)
            split_frames = split_frame_8_1_1(df, seed)
            split_counts = {
                split_name: len(split_df)
                for split_name, split_df in split_frames.items()
            }
            print(f"Seed {seed} split counts: {split_counts}")
            encoded = encode_split_frames(split_frames)
            in_channels = encoded["train"][0].x.shape[1]

            for set_config in selected_set_configs(dataset_config, args.set_configs):
                set_seed(seed)
                valid_sets = build_sets(encoded["valid"], set_config, shuffle=False)
                test_sets = build_sets(encoded["test"], set_config, shuffle=False)
                valid_loader = make_loader(valid_sets, args.batch_size, shuffle=False)
                test_loader = make_loader(test_sets, args.batch_size, shuffle=False)

                print(
                    f"\n{dataset_config.display_name} | seed={seed} | "
                    f"set={set_config.name}: valid_sets={len(valid_sets)}, "
                    f"test_sets={len(test_sets)}"
                )

                for model_config in selected_models:
                    print(f"Training {model_config.display_name}")
                    set_seed(seed)
                    model = model_config.factory(in_channels, args.hidden_dim).to(device)
                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=args.learning_rate,
                    )

                    run_dir = (
                        args.results_root
                        / dataset_config.dataset_id
                        / set_config.name
                        / f"seed_{seed}"
                        / model_config.model_id
                    )
                    run_dir.mkdir(parents=True, exist_ok=True)

                    prediction_csv_path = run_dir / "train_valid_test_predictions.csv"
                    epoch_metrics_csv_path = run_dir / "epoch_metrics.csv"
                    best_test_results_csv_path = run_dir / "test_results.csv"
                    for path in [
                        prediction_csv_path,
                        epoch_metrics_csv_path,
                        best_test_results_csv_path,
                    ]:
                        if path.exists():
                            path.unlink()

                    save_split_indices(split_frames, run_dir / "split_indices.csv")

                    best_val_total_loss = float("inf")
                    best_model_state = None
                    best_epoch = None

                    for epoch in range(1, args.epochs + 1):
                        set_seed(seed + epoch)
                        train_sets = build_sets(
                            encoded["train"],
                            set_config,
                            shuffle=True,
                        )
                        train_loader = make_loader(
                            train_sets,
                            args.batch_size,
                            balanced=True,
                        )

                        train_metrics, train_prediction_rows = train_epoch(
                            model,
                            train_loader,
                            optimizer,
                            device,
                            bce_loss_fn,
                            epoch=epoch,
                            collect_predictions=not args.skip_epoch_predictions,
                        )
                        valid_metrics, valid_prediction_rows = evaluate(
                            model,
                            valid_loader,
                            device,
                            bce_loss_fn,
                            phase="valid",
                            epoch=epoch,
                            collect_predictions=not args.skip_epoch_predictions,
                        )
                        test_metrics, test_prediction_rows = evaluate(
                            model,
                            test_loader,
                            device,
                            bce_loss_fn,
                            phase="test",
                            epoch=epoch,
                            collect_predictions=not args.skip_epoch_predictions,
                        )

                        if not args.skip_epoch_predictions:
                            append_prediction_rows(
                                prediction_csv_path,
                                train_prediction_rows
                                + valid_prediction_rows
                                + test_prediction_rows,
                            )

                        write_epoch_metrics(
                            epoch_metrics_csv_path,
                            {
                                "epoch": epoch,
                                "train_total_loss": train_metrics["total_loss"],
                                "train_set_bce": train_metrics["set_bce"],
                                "train_element_bce": train_metrics["element_bce"],
                                "valid_total_loss": valid_metrics["total_loss"],
                                "valid_set_bce": valid_metrics["set_bce"],
                                "valid_element_bce": valid_metrics["element_bce"],
                                "valid_set_auroc": valid_metrics["set_auroc"],
                                "valid_element_auroc": valid_metrics["element_auroc"],
                                "test_total_loss": test_metrics["total_loss"],
                                "test_set_bce": test_metrics["set_bce"],
                                "test_element_bce": test_metrics["element_bce"],
                                "test_set_auroc": test_metrics["set_auroc"],
                                "test_element_auroc": test_metrics["element_auroc"],
                            },
                        )

                        if valid_metrics["total_loss"] < best_val_total_loss:
                            best_val_total_loss = valid_metrics["total_loss"]
                            best_model_state = clone_state_dict(model)
                            best_epoch = epoch

                        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                            print(
                                f"Epoch {epoch}/{args.epochs} - "
                                f"train={train_metrics['total_loss']:.4f}, "
                                f"valid={valid_metrics['total_loss']:.4f}, "
                                f"valid_set_auc={valid_metrics['set_auroc']:.4f}, "
                                f"valid_element_auc={valid_metrics['element_auroc']:.4f}, "
                                f"test_set_auc={test_metrics['set_auroc']:.4f}, "
                                f"test_element_auc={test_metrics['element_auroc']:.4f}"
                            )

                    if best_model_state is None:
                        raise RuntimeError("No best model state was captured.")

                    model.load_state_dict(best_model_state)
                    torch.save(best_model_state, run_dir / "best_model.pt")
                    best_test_metrics, best_test_prediction_rows = evaluate(
                        model,
                        test_loader,
                        device,
                        bce_loss_fn,
                        phase="best_test",
                        epoch="best",
                        collect_predictions=True,
                    )
                    pd.DataFrame(best_test_prediction_rows).to_csv(
                        best_test_results_csv_path,
                        index=False,
                    )

                    summary_row = {
                        "dataset": dataset_config.dataset_id,
                        "dataset_display": dataset_config.display_name,
                        "set_config": set_config.name,
                        "set_size": set_config.table_label,
                        "seed": seed,
                        "model": model_config.model_id,
                        "model_display": model_config.display_name,
                        "best_epoch": best_epoch,
                        "best_val_total_loss": best_val_total_loss,
                        "test_total_loss": best_test_metrics["total_loss"],
                        "test_set_auroc": best_test_metrics["set_auroc"],
                        "test_element_auroc": best_test_metrics["element_auroc"],
                        "test_set_bce": best_test_metrics["set_bce"],
                        "test_element_bce": best_test_metrics["element_bce"],
                    }
                    summary_rows.append(summary_row)
                    pd.DataFrame([summary_row]).to_csv(
                        run_dir / "metrics_summary.csv",
                        index=False,
                    )

                    print(
                        f"Best epoch={best_epoch}, "
                        f"test_set_auc={best_test_metrics['set_auroc']:.4f}, "
                        f"test_element_auc={best_test_metrics['element_auroc']:.4f}"
                    )

    raw_path, table_csv_path, table_md_path = build_summary_tables(
        summary_rows,
        args.results_root,
    )
    print(f"\nSaved raw summary to {raw_path}")
    print(f"Saved table summary to {table_csv_path}")
    print(f"Saved markdown table to {table_md_path}")


def main():
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
