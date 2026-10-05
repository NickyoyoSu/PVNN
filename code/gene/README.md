# Genomic sequence learning (Section 6.4)

DNA sequence classification on the TEB benchmark (Khan et al., 2025). The training pipeline and the CNN / HCNN baselines are adapted from [HGE](https://github.com/rrkhan/HGE).

## Data

Download TEB as described in the HGE repository. Place the CSV files in `code/gene/data/`:

```
code/gene/data/train_<name>.csv
code/gene/data/valid_<name>.csv
code/gene/data/test_<name>.csv
```

`<name>` is one of `lines`, `sines`, `dna_hat_ac`, `processed_pseudogenes` or `unprocessed_pseudogenes`. Each sequence is one-hot encoded into 5 channels (`A/C/T/G/N`) and padded to `--length`.

## Run

```bash
# from the repository root
python code/gene/train.py -c code/gene/configs/PV_TEB.txt                 # PVCNN
python code/gene/train.py -c code/gene/configs/CNN_TEB.txt                # Euclidean CNN
python code/gene/train.py -c code/gene/configs/HCNN_SingleK_TEB.txt       # Lorentz HCNN, single curvature
```

Use `--dataset_name`, `--length` and `--k` to switch datasets. The best checkpoint on the validation split is evaluated on the test split at the end of training, as long as `output_dir` is set.

- `train.py`: training entry point
- `models/benchmark_models.py`: Euclidean, Lorentz, Poincaré and PV CNNs
- `utils/`: data loading, model and optimizer construction
- `configs/`: run configurations
