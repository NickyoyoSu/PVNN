# Proper Velocity Neural Networks (ICLR 2026)

Official PyTorch implementation of **Proper Velocity Neural Networks**.

Ziheng Chen\*, Zihan Su\*, Bernhard Schölkopf, Nicu Sebe (\* equal contribution)

[[OpenReview]](https://openreview.net/forum?id=UDIYU1X3vC) [[Poster]](assets/ICLR26-PVNN-Poster.pdf)

Most hyperbolic neural networks are built on the Poincaré ball or the hyperboloid. Both models are constrained, which can cause numerical problems near their boundaries. PVNN instead works in the Proper Velocity (PV) manifold, an unconstrained model of hyperbolic space. The paper derives the Riemannian operators of the PV space and uses them to build MLR, fully connected, convolutional, activation and batch normalization layers. These layers are evaluated on image classification, graph benchmarks and genomic sequence classification.

## Repository structure

```
assets/                   poster
code/
  stability/              numerical stability of PV vs Poincaré / Lorentz operators (Section 6.1)
  classification/         image classification with a PV MLR head      (Section 6.2)
  graph/                  node classification on graph benchmarks       (Section 6.3)
  gene/                   genomic sequence classification on TEB        (Section 6.4)
data/                     graph datasets: Disease, Airport, PubMed, Cora
lib/
  pv/                     PV manifold and PV layers (this paper)
  lorentz/ poincare/ klein/ Euclidean/    baseline geometries
  geoopt/                 vendored geoopt (manifold parameters, Riemannian optimizers)
  models/ utils/          ResNet backbones and training utilities
```

PV components in `lib/pv/`:

| Component | File |
| --- | --- |
| PV manifold: Riemannian and gyro operations | `manifold.py` (classification and genome models), `graph_ops.py` (graph models) |
| PV MLR | `graph_ops.py` (`PVManifoldMLR`), with the wrappers in `layers.py` and `mlr.py` |
| PV fully connected layer | `graph_ops.py` (`PVFC`), with the wrapper in `layers.py` |
| PV convolution, activation and batch normalization (1D) | `layers_1d.py`, `blocks_1d.py` |
| PV GyroBN | `gyrobn_pv.py` |

## Installation

```bash
conda create -n pvnn python=3.11
conda activate pvnn
pip install -r requirements.txt
```

Python 3.10 or newer is required. The code was tested with Python 3.11 and PyTorch 2.2. All commands below are run from the repository root.

## Numerical stability (Section 6.1)

```bash
python code/stability/numerical_test.py --device cpu
```

The script compares PV with the Poincaré ball and the Lorentz model in FP32 and FP64, plus FP16 on GPU. It reports:

- threshold sweeps for scalar multiplication r ⊗ x, giving failure rates as r grows
- one-shot scalar multiplication and addition diagnostics
- exp/log round-trip errors
- a precision sweep
- gradient magnitudes across radii

The options are `--kappa`, `--d` and `--batch`. The script uses the self-contained operator implementations in `code/stability/`.

## Image classification (Section 6.2)

A Euclidean ResNet-18 encoder is combined with different classification heads. CIFAR-10 and CIFAR-100 are downloaded automatically to `code/classification/data/`.

```bash
python code/classification/train.py -c code/classification/config/PV-ResNet18.txt --output_dir output/classification
```

| Config | Classification head |
| --- | --- |
| `PV-ResNet18.txt` | PV MLR, c = 0.15 (Appendix C.2) |
| `EP-ResNet18.txt` | Poincaré MLR. `mlr_type`: `b` Busemann (default), `g` Ganea et al. (2018), `hnn++` Shimizu et al. (2021) |
| `EL-ResNet18.txt` | Lorentz MLR (Bdeir et al., 2024) |
| `E-ResNet18.txt` | Euclidean linear layer |
| `L-ResNet18.txt` | Lorentz ResNet-18 encoder with a Lorentz MLR |

All configs train on CIFAR-100; use `--dataset CIFAR-10` to switch datasets. Command-line flags override config values, e.g. `--mlr_type g` with `EP-ResNet18.txt`.

To evaluate a trained model:

```bash
python code/classification/test.py -c code/classification/config/PV-ResNet18.txt \
    --load_checkpoint output/classification/best_PV-ResNet18.pth --mode test_accuracy
```

Tiny-ImageNet is also supported (`--dataset Tiny-ImageNet`). Extract `tiny-imagenet-200` into `code/classification/data/` to use it.

## Graph learning (Section 6.3)

The datasets are included in `data/`.

```bash
python code/graph/train.py
```

This trains PVNN on Disease, Airport, PubMed and Cora with 5 seeds using the hyperparameters in Appendix C.3, then prints the mean ± std test accuracy. The script takes no command-line arguments; edit `main()` in `code/graph/train.py` to change the setup:

- `datasets_to_run` selects the datasets.
- `model_types` selects the models: `pvnn` (ours); `hnn` and `hnn++` (Poincaré); `lnn` (Lorentz); `knn` (Klein); `fc` (Euclidean).
- `linear_type`, `inner_act` and `outer_act` (in `common_config` / `pvnn_extra`) select the PV layer variants used in the ablations.

Training runs on the CPU.

## Genomic sequence learning (Section 6.4)

Download the TEB datasets as described in the [HGE repository](https://github.com/rrkhan/HGE). Place `train_<name>.csv`, `valid_<name>.csv` and `test_<name>.csv` in `code/gene/data/`. The paper uses `lines`, `sines`, `dna_hat_ac`, `processed_pseudogenes` and `unprocessed_pseudogenes`.

```bash
python code/gene/train.py -c code/gene/configs/PV_TEB.txt
```

`PV_TEB.txt` trains PVCNN on hAT-Ac. PVCNN uses PV GyroBN and a single curvature shared by all layers (Appendix C.4). For other datasets, set `--dataset_name` and the maximum sequence length `--length`. For example, our SINEs run used `--dataset_name sines --length 500 --k 0.225`. The baseline configs `CNN_TEB.txt` (Euclidean CNN) and `HCNN_SingleK_TEB.txt` / `HCNN_MultiK_TEB.txt` (Lorentz HCNN) follow Khan et al. (2025).

## Citation

```bibtex
@inproceedings{chen2026proper,
  title     = {Proper Velocity Neural Networks},
  author    = {Chen, Ziheng and Su, Zihan and Sch{\"o}lkopf, Bernhard and Sebe, Nicu},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=UDIYU1X3vC}
}
```

## License

This project is released under the [MIT License](LICENSE). Third-party code keeps its original license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Acknowledgements

This code builds on [geoopt](https://github.com/geoopt/geoopt), [HyperbolicCV](https://github.com/kschwethelm/HyperbolicCV), [HGE](https://github.com/rrkhan/HGE) and [HGCN](https://github.com/HazyResearch/hgcn). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details and licenses.
