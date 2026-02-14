import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

WORKING_DIR = os.path.realpath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.realpath(os.path.join(WORKING_DIR, "..", ".."))
os.chdir(WORKING_DIR)

for path in (WORKING_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.append(path)

from lib.geoopt.optim import RiemannianAdam
from lib.data_loader import get_dataset
from models.geometric_models import build_model


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, train_loader, optimizer, criterion, epoch, print_interval=50):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if batch_idx % print_interval == 0:
            print(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)}]\t"
                f"Loss: {loss.item():.6f}"
            )


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100.0 * correct / len(test_loader.dataset)


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set on {model.model_type.upper()}: "
        f"Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    return accuracy


def run_single_experiment(model_type, config, train_loader, test_loader, input_dim, n_classes, seed):
    set_seed(seed)
    print(f"--- Training {model_type.upper()} with seed {seed} ---")

    model = build_model(
        model_type=model_type,
        dim=input_dim,
        hidden_dim=config["hidden_dim"],
        n_classes=n_classes,
        p_drop=config["dropout"],
        c=config["curvature"],
        inner_act=config.get("inner_act", "none"),
        outer_act=config.get("outer_act", "tangent"),
        linear_type=config.get("linear_type", "pvfc"),
    )

    optimizer = RiemannianAdam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, config["epochs"] + 1):
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            epoch,
            print_interval=config["print_interval"],
        )

        if epoch % 10 == 0 or epoch == config["epochs"]:
            current_accuracy = evaluate(model, test_loader)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                patience_counter = 0
                best_model_state = {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }
            else:
                patience_counter += 1

            if patience_counter >= config["patience"]:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {config['patience']} eval intervals)."
                )
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return test(model, test_loader, criterion)


def main():
    # Paper main graph experiment (Section 6.3 / Appendix C.3)
    model_types = ["pvnn"]
    datasets_to_run = ["disease", "airport", "pubmed", "cora"]

    common_config = {
        "epochs": 2000,
        "batch_size": 128,
        "weight_decay": 5e-4,
        "hidden_dim": 16,
        "num_runs": 5,
        "print_interval": 100,
        "patience": 200,
        "inner_act": "none",
        "outer_act": "tangent",
    }

    # Appendix C.3, Table 15
    dataset_overrides = {
        "disease": {"lr": 0.01, "dropout": 0.4, "curvature": 0.3},
        "airport": {"lr": 0.01, "dropout": 0.4, "curvature": 0.3},
        "pubmed": {"lr": 0.05, "dropout": 0.6, "curvature": 1.0},
        "cora": {"lr": 0.05, "dropout": 0.6, "curvature": 1.0},
    }

    pvnn_extra = {
        "linear_type": "pvfc",
        "outer_act": "tangent",
    }

    all_results = {}

    for dataset_name in datasets_to_run:
        print(f"\n========== Dataset: {dataset_name.upper()} ==========")

        shared_config = dict(common_config)
        shared_config.update(dataset_overrides[dataset_name])
        shared_config["dataset"] = dataset_name

        pvnn_config = dict(shared_config)
        pvnn_config.update(pvnn_extra)

        X_train, y_train, X_test, y_test = get_dataset(dataset_name)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(
            train_dataset, batch_size=shared_config["batch_size"], shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=shared_config["batch_size"], shuffle=False
        )

        input_dim = X_train.shape[1]
        n_classes = len(torch.unique(y_train))

        results = {model_type: [] for model_type in model_types}
        base_seed = 40

        for run in range(shared_config["num_runs"]):
            seed = base_seed + run
            print("\n=========================================================")
            print(f"--- Run {run + 1}/{shared_config['num_runs']} (seed: {seed}) ---")
            print("=========================================================")

            for model_type in model_types:
                print(f"\n--- Training and evaluating {model_type.upper()} ---")
                config = pvnn_config if model_type == "pvnn" else shared_config
                print(
                    f"config: lr={config['lr']}, dropout={config['dropout']}, "
                    f"weight_decay={config['weight_decay']}, curvature={config['curvature']}"
                )
                accuracy = run_single_experiment(
                    model_type,
                    config,
                    train_loader,
                    test_loader,
                    input_dim,
                    n_classes,
                    seed,
                )
                results[model_type].append(accuracy)

        all_results[dataset_name] = {
            "accuracies": results,
            "config": shared_config,
        }

    print("\n===== Final summary (mean ± std) =====")
    for dataset_name in datasets_to_run:
        print(f"\n{dataset_name.upper()}:")
        for model_type in model_types:
            accuracies = all_results[dataset_name]["accuracies"][model_type]
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            print(f"  {model_type.upper():<5}: {mean_acc:.2f}% ± {std_acc:.2f}%")


if __name__ == "__main__":
    main()
