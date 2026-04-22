import numpy as np
import pandas as pd
import torch

from .scaffold_split import scaffold_split
from graph_set_transformer.utils.graph_encoder import GraphEncoder

MOLECULENET_TASKS = {
    "bace": ["Class"],
    "bbbp": ["p_np"],
    "clintox": ["FDA_APPROVED", "CT_TOX"],
    "esol": ["ESOL predicted log solubility in mols per litre"],
    "freesolv": ["expt"],
    "hiv": ["HIV_active"],
    "lipo": ["exp"],
    "muv": [
        "MUV-692",
        "MUV-689",
        "MUV-846",
        "MUV-859",
        "MUV-644",
        "MUV-548",
        "MUV-852",
        "MUV-600",
        "MUV-810",
        "MUV-712",
        "MUV-737",
        "MUV-858",
        "MUV-713",
        "MUV-733",
        "MUV-652",
        "MUV-466",
        "MUV-832",
    ],
    "qm7": ["u0_atom"],
    "qm8": [
        "E1-CC2",
        "E2-CC2",
        "f1-CC2",
        "f2-CC2",
        "E1-PBE0",
        "E2-PBE0",
        "f1-PBE0",
        "f2-PBE0",
        "E1-CAM",
        "E2-CAM",
        "f1-CAM",
        "f2-CAM",
    ],
    "qm9": ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv"],
    "sider": [
        "Hepatobiliary disorders",
        "Metabolism and nutrition disorders",
        "Product issues",
        "Eye disorders",
        "Investigations",
        "Musculoskeletal and connective tissue disorders",
        "Gastrointestinal disorders",
        "Social circumstances",
        "Immune system disorders",
        "Reproductive system and breast disorders",
        "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
        "General disorders and administration site conditions",
        "Endocrine disorders",
        "Surgical and medical procedures",
        "Vascular disorders",
        "Blood and lymphatic system disorders",
        "Skin and subcutaneous tissue disorders",
        "Congenital, familial and genetic disorders",
        "Infections and infestations",
        "Respiratory, thoracic and mediastinal disorders",
        "Psychiatric disorders",
        "Renal and urinary disorders",
        "Pregnancy, puerperium and perinatal conditions",
        "Ear and labyrinth disorders",
        "Cardiac disorders",
        "Nervous system disorders",
        "Injury, poisoning and procedural complications",
    ],
    "tox21": [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ],
}


def molecule_net_task_loader(name: str, featurizer=None, **kwargs):
    return MOLECULENET_TASKS[name]


def from_df(df, smiles_column, y_columns):
    return (
        df[smiles_column].to_numpy(),
        df[y_columns].to_numpy(),
    )


def _split_with_official_subset(name, df):
    if name != "bace":
        raise ValueError(f"Official subset split is only supported for BACE, got {name}.")
    if "Model" not in df.columns:
        raise ValueError("Expected a 'Model' column for official subset splitting.")

    train = df[df["Model"] == "Train"].reset_index(drop=True)
    valid = df[df["Model"] == "Valid"].reset_index(drop=True)
    test = df[df["Model"] == "Test"].reset_index(drop=True)
    return train, valid, test


def molecule_net_loader(
    name,
    path,
    task_idx=0,
    featurizer=None,
    split_ratio=0.7,
    seed=42,
    task_name=None,
    split_mode="scaffold",
    y_columns=None,
    label_dtype=torch.float32,
    **kwargs,
):
    enc = GraphEncoder()

    df = pd.read_csv(path)

    if name in ["tox21"]:
        df = df.replace("", np.nan)
        df = df.dropna(subset=[task_name])

    if split_mode == "official":
        train, valid, test = _split_with_official_subset(name, df)
    else:
        train_ids, valid_ids, test_ids = scaffold_split(df, 0.1, 0.1, seed)
        train = df.loc[train_ids]
        valid = df.loc[valid_ids]
        test = df.loc[test_ids]

    tasks = y_columns if y_columns is not None else MOLECULENET_TASKS[name]

    train_smiles, train_y = from_df(train, "smiles", tasks)
    valid_smiles, valid_y = from_df(valid, "smiles", tasks)
    test_smiles, test_y = from_df(test, "smiles", tasks)

    if train_y.ndim == 1:
        train_y = np.expand_dims(train_y, -1)
        valid_y = np.expand_dims(valid_y, -1)
        test_y = np.expand_dims(test_y, -1)

    if y_columns is None:
        train_y = np.array(train_y[:, task_idx])
        valid_y = np.array(valid_y[:, task_idx])
        test_y = np.array(test_y[:, task_idx])
    else:
        train_y = np.array(train_y, dtype=np.float32)
        valid_y = np.array(valid_y, dtype=np.float32)
        test_y = np.array(test_y, dtype=np.float32)

    print(
        f"Using {name} split mode '{split_mode}': "
        f"Train={len(train)}, Valid={len(valid)}, Test={len(test)}"
    )

    print("Encoding training set ...")
    train_dataset = enc.encode(train_smiles, train_y, label_dtype=label_dtype)
    print("Encoding validation set ...")
    valid_dataset = enc.encode(valid_smiles, valid_y, label_dtype=label_dtype)
    print("Encoding test set ...")
    test_dataset = enc.encode(test_smiles, test_y, label_dtype=label_dtype)

    return train_dataset, valid_dataset, test_dataset, tasks


def get_class_weights(y, task_idx=None):
    if task_idx is None:
        _, counts = np.unique(y, return_counts=True)
        weights = [1 - c / y.shape[0] for c in counts]

        return np.array(weights), np.array(counts)

    y_t = y.T

    _, counts = np.unique(y_t[task_idx], return_counts=True)
    weights = [1 - c / y_t[task_idx].shape[0] for c in counts]

    return np.array(weights), np.array(counts)
