# Graph learning (Section 6.3)

Node classification on Disease, Airport, PubMed and Cora. Node features are fed to the model as i.i.d. samples, without using the graph structure (Appendix C.3).

```bash
# from the repository root
python code/graph/train.py
```

- `train.py` is the training entry point. It holds the experiment setup in `main()`: datasets, models, the per-dataset hyperparameters of Appendix C.3, and the PV layer variants.
- `models/geometric_models.py` defines the two-layer models: `pvnn`, `hnn`, `hnn++`, `lnn`, `knn` and `fc`.
- The data loaders are in `lib/data_loader.py`. The datasets are in the repository-level `data/` folder.
