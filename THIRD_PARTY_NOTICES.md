# Third-party code

This repository includes code adapted from the projects below. We thank the authors for releasing their code.

| Component in this repository | Source | License |
| --- | --- | --- |
| `lib/geoopt/` (vendored geoopt 0.5.0) | [geoopt/geoopt](https://github.com/geoopt/geoopt) | Apache License 2.0, see `lib/geoopt/LICENSE` |
| `code/classification/`, `lib/lorentz/`, `lib/models/`, `lib/Euclidean/`, `lib/poincare/layers/`, `lib/utils/` | [kschwethelm/HyperbolicCV](https://github.com/kschwethelm/HyperbolicCV) (Bdeir et al., ICLR 2024) | MIT, reproduced below |
| `code/gene/` (TEB training pipeline and CNN / HCNN baselines) | [rrkhan/HGE](https://github.com/rrkhan/HGE) (Khan et al., ICLR 2025) | no license file in the upstream repository |
| `data/` graph datasets, graph data loaders in `lib/data_loader.py`, Poincaré ball operations in `lib/poincare/hnn_manifold.py` | [HazyResearch/hgcn](https://github.com/HazyResearch/hgcn) (Chami et al., NeurIPS 2019) | no license file in the upstream repository |

## HyperbolicCV license

```
MIT License

Copyright (c) 2023 Kristian Schwethelm

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
