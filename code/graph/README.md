# Graph Learning (PVNN) Experiments

This folder is for the Section 6.3 graph learning main experiments.

## Structure

- `configs/` baseline configs extracted from the original experiments
- `../data/` dataset root (place raw files here)
- `lib/` data loaders and utilities
- `models/` manifolds and model definitions
- `train.py` main training entry

## Data layout

Create subfolders under the repo root `data/`:

- `data/cora/`
- `data/pubmed/`
- `data/airport/`
- `data/disease_nc/` and `data/disease_lp/`

The file formats are the same as the original loader expects.

## Run

From `graph/`:

```bash
python train.py
```

Adjust datasets and flags inside `train.py` as needed.
