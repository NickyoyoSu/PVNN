# Gene Package for HGE 6.4 TEB Experiments

This directory reorganizes the runnable code for the paper section 6.4 graph learning and ablation experiments into a delivery-ready layout.

## Layout

- `code/gene/train.py`: training entrypoint.
- `code/gene/models/`: model definitions.
- `lib/`: shared manifold and geometry operators (PV, Lorentz, Poincare, Euclidean, geoopt).
- `code/gene/utils/`: dataset loading and initialization utilities.
- `code/gene/configs/`: baseline configs.
- `code/gene/data/`: placeholder dataset directory. Put TEB CSV files here.

## Data Convention

- Expected input tensor format in training: `[B, C, L]`.
- Current data loader encodes DNA bases into 5 channels (`A/C/T/G/N`) and returns shape `[B, 5, L]`.
- Sequence length `L` should follow the dataset setting in config (500 or 1000 for TEB tasks).

## Run

From repository root:

```bash
python code/gene/train.py -c code/gene/configs/HCNN_SingleK_TEB.txt
```

Before running, place files like `train_lines.csv`, `valid_lines.csv`, `test_lines.csv` in `code/gene/data/`.
