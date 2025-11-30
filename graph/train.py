from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from geoopt.optim import RiemannianAdam
from torch.utils.data import DataLoader, TensorDataset

from configs import DatasetConfig, ExperimentConfig, apply_overrides
from data.loader import get_dataset
from models.geometric_models import build_model, build_pvnn_frechet_sweep


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_epoch(model, loader, optimizer, criterion, epoch, print_interval: int = 50):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if print_interval > 0 and batch_idx % print_interval == 0:
            size = len(loader.dataset)
            processed = batch_idx * len(data)
            print(f"Train Epoch: {epoch} [{processed}/{size}]\tLoss: {loss.item():.6f}")


def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100.0 * correct / len(loader.dataset)
    return accuracy


def test(model, loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)
    print(
        f"\nTest set on {getattr(model, 'model_type', 'model').upper()}: "
        f"Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({accuracy:.2f}%)\n"
    )
    return accuracy


def train_model_instance(model, cfg: DatasetConfig, train_loader, test_loader, seed: int):
    set_seed(seed)
    if getattr(model, "model_type", "").lower() != "fc":
        optimizer = RiemannianAdam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    patience_counter = 0
    best_state = None

    print_interval = getattr(cfg, "print_interval", 50)
    for epoch in range(1, cfg.epochs + 1):
        train_epoch(model, train_loader, optimizer, criterion, epoch, print_interval)
        if epoch % 10 == 0 or epoch == cfg.epochs:
            current_accuracy = evaluate(model, test_loader)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping at epoch {epoch} - no improvement for {cfg.patience} eval steps")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return test(model, test_loader, criterion)


def _build_loaders(data_tensors, batch_size: int):
    x_train, y_train, x_test, y_test = data_tensors
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, x_train.shape[1], len(torch.unique(y_train))


def _prepare_pvnn_models(cfg: DatasetConfig, input_dim: int, n_classes: int, frechet_iters):
    if not frechet_iters:
        model = build_model(
            model_type="pvnn",
            dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            n_classes=n_classes,
            p_drop=cfg.dropout,
            c=cfg.curvature,
            inner_act=cfg.inner_act,
            outer_act=cfg.outer_act,
        )
        return {None: model}

    iters_list = []
    for it in frechet_iters:
        if isinstance(it, str) and it.lower() in {"inf", "infinite"}:
            iters_list.append("inf")
        else:
            iters_list.append(int(it))

    models = build_pvnn_frechet_sweep(
        dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        n_classes=n_classes,
        p_drop=cfg.dropout,
        c=cfg.curvature,
        iters_list=iters_list,
        inner_act=cfg.inner_act,
        outer_act=cfg.outer_act,
        bn_mode="gyro",
    )
    return models


def run_experiments(
    exp_cfg: ExperimentConfig,
    dataset_configs: Dict[str, DatasetConfig],
    data_root,
) -> Dict[str, Dict]:
    summary = {}
    datasets = [name.lower() for name in exp_cfg.datasets]
    frechet_iters = exp_cfg.frechet_iters or []

    for dataset_name in datasets:
        data_tensors = get_dataset(dataset_name, data_root)
        ds_cfg = dataset_configs.get(dataset_name)
        if ds_cfg is None:
            raise ValueError(f"No preset found for dataset {dataset_name}.")
        ds_cfg = apply_overrides(ds_cfg, exp_cfg)

        train_loader, test_loader, input_dim, n_classes = _build_loaders(
            data_tensors,
            ds_cfg.batch_size,
        )

        summary[dataset_name] = {"runs": defaultdict(list), "config": asdict(ds_cfg)}

        for run_idx in range(exp_cfg.runs):
            seed = exp_cfg.base_seed + run_idx
            for model_type in exp_cfg.model_types:
                if model_type == "pvnn":
                    models = _prepare_pvnn_models(ds_cfg, input_dim, n_classes, frechet_iters)
                    for iter_key, model in models.items():
                        label = "pvnn" if iter_key is None else f"pvnn_iter={'inf' if iter_key == -1 else iter_key}"
                        print(f"\n--- Training and evaluating {label.upper()} (seed={seed}) ---")
                        acc = train_model_instance(model, ds_cfg, train_loader, test_loader, seed)
                        summary[dataset_name]["runs"][label].append({"seed": seed, "acc": acc})
                else:
                    print(f"\n--- Training and evaluating {model_type.upper()} (seed={seed}) ---")
                    model = build_model(
                        model_type=model_type,
                        dim=input_dim,
                        hidden_dim=ds_cfg.hidden_dim,
                        n_classes=n_classes,
                        p_drop=ds_cfg.dropout,
                        c=ds_cfg.curvature,
                        inner_act=ds_cfg.inner_act,
                        outer_act=ds_cfg.outer_act,
                    )
                    acc = train_model_instance(model, ds_cfg, train_loader, test_loader, seed)
                    summary[dataset_name]["runs"][model_type].append({"seed": seed, "acc": acc})

        summary[dataset_name]["runs"] = dict(summary[dataset_name]["runs"])

    return summary

