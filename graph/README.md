# Proper Velocity NN - Section 6.2 Reproduction

This folder contains a cleaned-up implementation of the experiments described in Section 6.2 of *Proper Velocity Neural Networks* (ICLR under review). Besides the PVNN architecture (with the gyro-BN variant used in the paper), the release bundles additional hyperbolic baselines: Poincare HNN, HNN++, Lorentz (hyperboloid), Klein, and a Euclidean FC control.

## Folder Layout

```
graph/
|-- lib/                    # Geometry + layer primitives grouped by manifold
|-- data/                   # Dataset loaders (code only)
|-- models/                 # Model factory (PVNN + baselines)
|-- configs/                # Experiment/datset configs
|-- train.py                # Training orchestration
|-- main.py                 # CLI entry point
|-- requirements.txt
|-- pyproject.toml
\`-- README.md
```

## Quick Start

1. **Install dependencies**  
   ```bash
   cd graph
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Prepare data**  
   Place the datasets under `/Users/nick/Documents/czh/code/data` (default) or any folder and pass the location through `--data-root`. The loaders expect the same raw files as the original PVNN repository (Cora/PubMed citation splits, Disease, Airport). The `data/` package only contains loader code; point `--data-root` to your actual storage (or symlink `data/raw` to it if you prefer).

3. **Run experiments**  
   ```bash
   PYTHONPATH=. python main.py \
       --datasets cora pubmed disease airport \
       --data-root /Users/nick/Documents/czh/code/data \
       --model-types pvnn hnn hnn++ lnn knn fc \
       --runs 5
   ```

## Configuration

Dataset-specific hyperparameters mirror Table 6.2 in the paper and are stored in `configs/datasets.yaml`. Override any value via CLI flags, e.g.:

```bash
PYTHONPATH=. python main.py --datasets cora --epochs 800 --lr 0.01
```

## Reproducibility Notes

- Seeds are set per run for backbone, CUDA, and NumPy.
- `--model-types` accepts any combination of `pvnn`, `hnn`, `hnn++`, `lnn`, `knn`, `fc`. When multiple models are supplied, the script trains each one per dataset/seed and reports a grouped summary.
- The default `--data-root` is `/Users/nick/Documents/czh/code/data`. Override via CLI if you keep datasets elsewhere.
- The PVNN model uses two PV fully connected blocks with an optional GyroBN layer between them and the PV manifold MLR head.
- The implementation exposes `--time-test` to match the time-measurement setting reported in the paper.

For details on the derivations behind PVFC and the gyro batch-normalization strategy, please refer to the original manuscript.
