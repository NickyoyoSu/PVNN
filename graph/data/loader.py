from __future__ import annotations

import os
import pickle as pkl
from pathlib import Path
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split


def _normalize(mx: np.ndarray | sp.spmatrix) -> np.ndarray:
    """Row-normalize a (possibly sparse) matrix."""

    if sp.isspmatrix(mx):
        mx = np.array(mx.todense())
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def _bin_feat(feat: np.ndarray, bins) -> np.ndarray:
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


def _sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.sparse.FloatTensor:
    """Convert a scipy sparse matrix to a torch sparse tensor."""

    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def _load_data_airport(dataset_str: str, data_path: Path, return_label: bool = False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + ".p"), "rb"))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.nodes[u]["feat"] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = _bin_feat(labels, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7])
        return sp.csr_matrix(adj), features, labels
    return sp.csr_matrix(adj), features


def _parse_index_file(filename: Path):
    index = []
    for line in filename.open():
        index.append(int(line.strip()))
    return index


def _load_citation_data_nc(dataset_str: str, use_feats: bool, data_path: Path, split_seed: int | None = None):
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for name in names:
        with open(os.path.join(data_path, f"ind.{dataset_str}.{name}"), "rb") as f:
            if os.sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding="latin1"))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = _parse_index_file(Path(os.path.join(data_path, f"ind.{dataset_str}.test.index")))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test


def _load_citation_data_clf(dataset: str, data_path: Path):
    """Simplified loader used for our pure-feature MLP-style experiments."""

    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for name in names:
        with open(data_path / f"ind.{dataset}.{name}", "rb") as f:
            objects.append(pkl.load(f, encoding="latin1"))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = _parse_index_file(data_path / f"ind.{dataset}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    return features, labels, idx_train, idx_test


def get_cora_dataset(data_path: Path):
    """Exactly mirrors original get_cora_dataset from data_loader.py."""

    print("Loading and preprocessing Cora...")
    features, labels, idx_train, idx_test = _load_citation_data_clf("cora", data_path)
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    features = _normalize(features)
    x = torch.FloatTensor(features)
    y = torch.LongTensor(labels)
    x_train = x[idx_train]
    y_train = y[idx_train]
    x_test = x[idx_test]
    y_test = y[idx_test]
    print(f"Dataset ready: {len(x)} samples, {x.shape[1]} features, {len(torch.unique(y))} classes.")
    print(f"Train split: {len(x_train)} samples, test split: {len(x_test)} samples.")
    return x_train, y_train, x_test, y_test


def get_pubmed_dataset(data_path: Path):
    """Exactly mirrors original get_pubmed_dataset from data_loader.py."""

    print("Loading and preprocessing Pubmed...")
    features, labels, idx_train, idx_test = _load_citation_data_clf("pubmed", data_path)
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    features = _normalize(features)
    x = torch.FloatTensor(features)
    y = torch.LongTensor(labels)
    x_train = x[idx_train]
    y_train = y[idx_train]
    x_test = x[idx_test]
    y_test = y[idx_test]
    print(f"Dataset ready: {len(x)} samples, {x.shape[1]} features, {len(torch.unique(y))} classes.")
    print(f"Train split: {len(x_train)} samples, test split: {len(x_test)} samples.")
    return x_train, y_train, x_test, y_test


def get_disease_dataset(data_path: Path, variant: str = "nc", test_size: float = 0.2, random_state: int = 42):
    """Port of original get_disease_dataset from data_loader.py."""

    import pandas as pd
    from scipy.sparse import load_npz

    folder = f"disease_{variant}"
    full_path = os.path.join(str(data_path), folder)

    features_path = os.path.join(full_path, f"disease_{variant}.feats.npz")
    try:
        features = load_npz(features_path)
        if sp.isspmatrix(features):
            features = np.array(features.todense())
    except Exception as e:
        print(f"Failed to load feature file: {e}")
        try:
            features = np.load(features_path, allow_pickle=True)
        except Exception:
            raise FileNotFoundError(f"Failed to load feature file: {features_path}")

    labels_path = os.path.join(full_path, f"disease_{variant}.labels.npy")
    try:
        labels = np.load(labels_path)
    except Exception as e:
        print(f"Failed to load label file: {e}")
        raise FileNotFoundError(f"Failed to load label file: {labels_path}")

    edges_path = os.path.join(full_path, f"disease_{variant}.edges.csv")
    try:
        edges = pd.read_csv(edges_path, header=None)
        print(f"Loaded {len(edges)} edges")
    except Exception as e:
        print(f"Failed to load edge file or file empty: {e}")

    features = _normalize(features)
    x = torch.FloatTensor(features)
    y = torch.LongTensor(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    num_classes = len(torch.unique(y))
    print(f"Dataset ready: {len(x)} samples, {x.shape[1]} features, {num_classes} classes.")
    print(f"Train split: {len(x_train)} samples, test split: {len(x_test)} samples.")
    print(f"Class histogram: {torch.bincount(y)}")
    return x_train, y_train, x_test, y_test


def get_airport_dataset(data_path: Path, test_size: float = 0.2, random_state: int = 42):
    """Port of original get_airport_dataset from data_loader.py."""

    from sklearn.preprocessing import StandardScaler

    print("Loading and preprocessing the airport dataset...")
    adj, features, labels = _load_data_airport("airport", data_path, return_label=True)

    print("Raw feature range:", np.min(features), np.max(features))
    extreme_mask = np.abs(features) > 100
    if extreme_mask.any():
        print(f"Detected {extreme_mask.sum()} extreme values; clipped.")
        features = np.clip(features, -100, 100)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print("Scaled feature range:", np.min(features_scaled), np.max(features_scaled))

    labels_binary = (labels > np.median(labels)).astype(int)

    x = torch.FloatTensor(features_scaled)
    y = torch.LongTensor(labels_binary)

    x = torch.clamp(x, -10, 10)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Dataset ready: {len(x)} samples, {x.shape[1]} features, {len(torch.unique(y))} classes.")
    print(f"Train split: {len(x_train)} samples, test split: {len(x_test)} samples.")
    return x_train, y_train, x_test, y_test


def get_dataset(dataset_name: str, data_root: Path):
    """Main entry point used by Section 6.2 code (standalone version)."""

    name = dataset_name.lower()
    if name == "airport":
        return get_airport_dataset(data_path=data_root / "airport")
    if name == "cora":
        return get_cora_dataset(data_path=data_root / "cora")
    if name == "pubmed":
        return get_pubmed_dataset(data_path=data_root / "pubmed")
    if name in ("disease_nc", "disease"):
        return get_disease_dataset(variant="nc", data_path=data_root)
    if name == "disease_lp":
        return get_disease_dataset(variant="lp", data_path=data_root)
    raise ValueError(f"Unknown dataset: {dataset_name}")


