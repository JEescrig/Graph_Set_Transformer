import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Batch


def _extract_graph_targets(data):
    values = data.y.view(-1)
    class_label = int(values[0].item())
    regression_target = float(values[1].item()) if values.numel() > 1 else float(values[0].item())
    return class_label, regression_target


def _extract_set_label(set_item):
    _, metadata = set_item
    if isinstance(metadata, dict):
        return int(metadata["set_label"])
    return int(metadata)


class SetDataset(Dataset):
    def __init__(self, sets):
        self.sets = sets

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        return self.sets[idx]


class BalancedSetBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_classes=2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.class_indices = {i: [] for i in range(num_classes)}
        for idx, set_item in enumerate(dataset):
            self.class_indices[_extract_set_label(set_item)].append(idx)

        self.sets_per_class = batch_size // num_classes
        min_class_count = min(len(indices) for indices in self.class_indices.values())
        self.num_batches = min_class_count // self.sets_per_class

    def __iter__(self):
        shuffled_indices = {
            class_id: np.random.permutation(indices).tolist()
            for class_id, indices in self.class_indices.items()
        }

        batches = []
        for batch_idx in range(self.num_batches):
            batch = []
            for class_id in range(self.num_classes):
                start = batch_idx * self.sets_per_class
                end = start + self.sets_per_class
                batch.extend(shuffled_indices[class_id][start:end])

            np.random.shuffle(batch)
            batches.append(batch)

        np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return self.num_batches


def collate_sets(batch_of_sets, verbose=False):
    all_graphs = []
    set_assignments = []
    set_labels = []
    graph_regression_targets = []
    class_counts = {}

    for set_idx, (graph_set, metadata) in enumerate(batch_of_sets):
        set_label = _extract_set_label((graph_set, metadata))
        set_labels.append(set_label)
        class_counts[set_label] = class_counts.get(set_label, 0) + 1

        for graph in graph_set:
            _, regression_target = _extract_graph_targets(graph)
            all_graphs.append(graph)
            set_assignments.append(set_idx)
            graph_regression_targets.append(regression_target)

    if verbose:
        print(f"Batch class distribution: {class_counts}")

    return (
        Batch.from_data_list(all_graphs),
        torch.tensor(set_assignments, dtype=torch.long),
        torch.tensor(set_labels, dtype=torch.long),
        torch.tensor(graph_regression_targets, dtype=torch.float32),
    )


def make_label_homogeneous_sets(dataset, set_size, shuffle=False):
    label_groups = defaultdict(list)
    for data in dataset:
        label, _ = _extract_graph_targets(data)
        label_groups[label].append(data)

    sets = []
    for label, graphs in label_groups.items():
        n = len(graphs)
        for i in range(n):
            current_set = [graphs[i]]
            other_indices = [j for j in range(n) if j != i]

            if set_size > 1:
                if not other_indices:
                    sampled_indices = []
                elif len(other_indices) >= set_size - 1:
                    sampled_indices = random.sample(other_indices, set_size - 1)
                else:
                    sampled_indices = random.choices(other_indices, k=set_size - 1)
                current_set.extend([graphs[j] for j in sampled_indices])

            sets.append((current_set, {"set_label": label, "set_size": len(current_set)}))

    if shuffle:
        random.shuffle(sets)

    return sets


def make_label_homogeneous_sets_rand_card(dataset, min_size=1, max_size=10, shuffle=True):
    label_groups = defaultdict(list)
    for data in dataset:
        label, _ = _extract_graph_targets(data)
        label_groups[label].append(data)

    sets = []
    for label, graphs in label_groups.items():
        n = len(graphs)
        for i in range(n):
            current_set_size = random.randint(min_size, max_size)
            current_set = [graphs[i]]
            other_indices = [j for j in range(n) if j != i]

            if current_set_size > 1:
                if not other_indices:
                    sampled_indices = []
                elif len(other_indices) >= current_set_size - 1:
                    sampled_indices = random.sample(other_indices, current_set_size - 1)
                else:
                    sampled_indices = random.choices(other_indices, k=current_set_size - 1)
                current_set.extend([graphs[j] for j in sampled_indices])

            sets.append(
                (
                    current_set,
                    {"set_label": label, "set_size": len(current_set)},
                )
            )

    if shuffle:
        random.shuffle(sets)

    return sets
