import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset

from torch_geometric.data import Batch


class SetDataset(Dataset):
    def __init__(self, sets):
        self.sets = sets

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        return self.sets[idx]


def collate_sets(batch_of_sets):
    all_graphs = []
    set_assignments = []
    labels = []

    for set_idx, (graph_set, label) in enumerate(batch_of_sets):
        all_graphs.extend(graph_set)
        set_assignments.extend([set_idx] * len(graph_set))
        labels.append(label)

    return (
        Batch.from_data_list(all_graphs),
        torch.tensor(set_assignments, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def make_label_homogeneous_sets(dataset, set_size):
    # Group by label
    label_groups = defaultdict(list)
    for data in dataset:
        label_groups[int(data.y.item())].append(data)

    sets = []

    for label, graphs in label_groups.items():
        random.shuffle(graphs)
        for i in range(0, len(graphs), set_size):
            sets.append((graphs[i : i + set_size], label))

    random.shuffle(sets)
    return sets
