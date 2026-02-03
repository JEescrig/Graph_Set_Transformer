import numpy as np
from tdc.single_pred import ADME
from graph_set_transformer.utils.graph_encoder import GraphEncoder


def from_df(df, smiles_column, y_columns):
    return (
        df[smiles_column].to_numpy(),
        df[df.columns.intersection(y_columns)].to_numpy(),
    )


def tdc_adme_task_loader(name: str, featurizer=None, **kwargs):
    # return ["Solubility_AqSolDB"]
    # return ["Lipophilicity_AstraZeneca"]
    # return ["PPBR_AZ"]
    # return ["Bioavailability_Ma"]
    # return ["CYP2C9_Veith"]
    return ["BBB_Martins"]


def tdc_adme_loader(name: str, featurizer=None, seed=42, **kwargs):
    enc = GraphEncoder()

    # task_name = kwargs.get("task_name", None)
    # print(task_name)

    data = ADME(name=name)
    split = data.get_split(method="scaffold")

    tasks = ["Y"]

    train_smiles, train_y = from_df(split["train"], "Drug", tasks)
    valid_smiles, valid_y = from_df(split["valid"], "Drug", tasks)
    test_smiles, test_y = from_df(split["test"], "Drug", tasks)

    if len(train_y.shape) == 1:
        train_y = np.expand_dims(train_y, -1)
        valid_y = np.expand_dims(valid_y, -1)
        test_y = np.expand_dims(test_y, -1)

    train_y = np.array(train_y[:, 0])
    valid_y = np.array(valid_y[:, 0])
    test_y = np.array(test_y[:, 0])

    print("Encoding training set ...")
    train_dataset = enc.encode(train_smiles, train_y)
    print("Encoding validation set ...")
    valid_dataset = enc.encode(valid_smiles, valid_y)
    print("Encoding test set ...")
    test_dataset = enc.encode(test_smiles, test_y)

    return train_dataset, valid_dataset, test_dataset, tasks
