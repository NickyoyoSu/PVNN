import torch
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import pickle as pkl
import networkx as nx
import os
import sys


_DATA_ROOT_CANDIDATES = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code", "graph", "data")),
]
DATA_ROOT = next((p for p in _DATA_ROOT_CANDIDATES if os.path.isdir(p)), _DATA_ROOT_CANDIDATES[0])



def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()

def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features

def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features
    


def get_disease_dataset(variant='nc', data_path=DATA_ROOT, test_size=0.2, random_state=42):
    import pandas as pd
    from scipy.sparse import load_npz
    
    folder = f"disease_{variant}"
    print(f"Loading and processing Disease {variant.upper()} dataset...")
    
    full_path = os.path.join(data_path, folder)
    
    features_path = os.path.join(full_path, f"disease_{variant}.feats.npz")
    try:
        features = load_npz(features_path)
        if sp.isspmatrix(features):
            features = np.array(features.todense())
    except Exception as e:
        print(f"Failed to load feature file: {e}")
        try:
            features = np.load(features_path, allow_pickle=True)
        except:
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
        print(f"Successfully loaded edge data: {len(edges)} edges")
    except Exception as e:
        print(f"Failed to load edge file or file is empty: {e}")
    
    features = normalize(features)                    
    
    X = torch.FloatTensor(features)
    y = torch.LongTensor(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    num_classes = len(torch.unique(y))
    print(f"Dataset ready: {len(X)} samples, {X.shape[1]} features, {num_classes} classes.")
    print(f"Train split: {len(X_train)} samples, test split: {len(X_test)} samples.")
    print(f"Class distribution: {torch.bincount(y)}")
    
    return X_train, y_train, X_test, y_test
    
def get_pubmed_dataset(data_path=os.path.join(DATA_ROOT, "pubmed")):
    print("Loading and processing PubMed dataset...")
    
    features, labels, idx_train, idx_test = load_citation_data('pubmed', True, data_path)

    if sp.isspmatrix(features):
        features = np.array(features.todense())
    features = normalize(features)                      

    X = torch.FloatTensor(features)
    y = torch.LongTensor(labels)

    X_train = X[idx_train]
    y_train = y[idx_train]
    X_test = X[idx_test]
    y_test = y[idx_test]

    print(f"Dataset ready: {len(X)} samples, {X.shape[1]} features, {len(torch.unique(y))} classes.")
    print(f"Train split: {len(X_train)} samples, test split: {len(X_test)} samples.")
    
    return X_train, y_train, X_test, y_test



def get_airport_dataset(data_path=os.path.join(DATA_ROOT, "airport"), test_size=0.2, random_state=42):
    print("Loading and processing Airport dataset...")
    
    adj, features, labels = load_data_airport('airport', data_path, return_label=True)
    
    from sklearn.preprocessing import StandardScaler
    
    print("Original feature range:", np.min(features), np.max(features))
    extreme_mask = np.abs(features) > 100
    if extreme_mask.any():
        print(f"Detected {extreme_mask.sum()} extreme values; clipped them.")
        features = np.clip(features, -100, 100)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print("Feature range after standardization:", np.min(features_scaled), np.max(features_scaled))

    labels_binary = (labels > np.median(labels)).astype(int)

    X = torch.FloatTensor(features_scaled)
    y = torch.LongTensor(labels_binary)
    
    X = torch.clamp(X, -10, 10)               
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Dataset ready: {len(X)} samples, {X.shape[1]} features, {len(torch.unique(y))} classes.")
    print(f"Train split: {len(X_train)} samples, test split: {len(X_test)} samples.")
    
    return X_train, y_train, X_test, y_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))

    if not use_feats:
        features = sp.eye(features.shape[0])
    return features, labels, idx_train, idx_test

def get_cora_dataset(data_path=os.path.join(DATA_ROOT, "cora")):
    print("Loading and processing Cora dataset...")
    
    features, labels, idx_train, idx_test = load_citation_data('cora', True, data_path)

    if sp.isspmatrix(features):
        features = np.array(features.todense())
    features = normalize(features)                     

    X = torch.FloatTensor(features)
    y = torch.LongTensor(labels)

    X_train = X[idx_train]
    y_train = y[idx_train]
    X_test = X[idx_test]
    y_test = y[idx_test]

    print(f"Dataset ready: {len(X)} samples, {X.shape[1]} features, {len(torch.unique(y))} classes.")
    print(f"Train split: {len(X_train)} samples, test split: {len(X_test)} samples.")
    
    return X_train, y_train, X_test, y_test


def get_dataset(dataset_name, data_path_root=DATA_ROOT):
    if dataset_name.lower() == 'airport':
        return get_airport_dataset(data_path=os.path.join(data_path_root, 'airport'))
    elif dataset_name.lower() == 'cora':
        return get_cora_dataset(data_path=os.path.join(data_path_root, 'cora'))
    elif dataset_name.lower() == 'pubmed':
        return get_pubmed_dataset(data_path=os.path.join(data_path_root, 'pubmed'))
    elif dataset_name.lower() == 'disease_nc' or dataset_name.lower() == 'disease':
        return get_disease_dataset(variant='nc', data_path=data_path_root)
    elif dataset_name.lower() == 'disease_lp':
        return get_disease_dataset(variant='lp', data_path=data_path_root)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


if __name__ == '__main__':
    print("--- Testing Airport data loader ---")
    X_train_air, y_train_air, X_test_air, y_test_air = get_dataset('airport')
    print("X_train shape:", X_train_air.shape)
    print("y_train shape:", y_train_air.shape)
    print("y_train class distribution:", torch.bincount(y_train_air))

    print("\n--- Testing Cora data loader ---")
    X_train_cora, y_train_cora, X_test_cora, y_test_cora = get_dataset('cora')
    print("X_train shape:", X_train_cora.shape)
    print("y_train shape:", y_train_cora.shape)
    print("y_train class distribution:", torch.bincount(y_train_cora))
