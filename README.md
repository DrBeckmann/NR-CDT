# NR-CDT

[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

This repository contains the code and experiments for the paper:

1. Matthias Beckmann, Robert Beinert, Jonas Bresch,
   '[Max-Normalized Radon Cumulative Distribution
   Transform for Limited Data Classification](https://doi.org/10.48550/arXiv.2411.16282)',
   [arXiv:2411.16282](https://arxiv.org/abs/2411.16282)
   (2025):
   Accepted for publication in
   *Proceedings of SSVM 2025*,
   10th International Conference on
   Scale Space and Variational Methods in Computer Vision,
   May 18-22 2025,
   Totnes, Devon, UK.

Please cite the paper if you use the code.

## Usage

To initialize the environment for the experiments,
start a Julia REPL in the main directory
and use the following commands:

```julia
import Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Examples

The basic usage of the code is shown
in the following Pluto notebooks:

- `examples/Templates.jl`:
  Generation of simple shapes for academic templates.
- `examples/ImageTransformations.jl`
  Generation of affinely transformed datasets.
- `examples/RadonCDT.jl`:
  Calculation of the max-normalized RCDT.

To run Pluto notebooks,
open a Julia REPL in the main directory
and start Pluto via

```julia
using Pluto; Pluto.run()
```

## Experiments

The simulations in [1] can be reproduced
using the following Pluto notebooks:

- `ssvm2025/Tab2-academic.jl`:
  Nearest neighbour classification for academic dataset.
  This experiment corresponds to
  Table 2 (left) and Figure 4.
- `ssvm2025/Tab2-mnist.jl`:
  Nearest neighbour classification for LinMNIST.
  This experiment corresponds to
  Table 2 (right).
- `ssvm2025/Tab3.jl`:
  Linear SVM classification for academic dataset.
  This experiment corresponds to
  Table 3.
- `ssvm2025/Tab4.jl`:
  Linear SVM classification for LinMNIST.
  This experiment corresponds to
  Table 4.

## License

This code is licensed under a MIT License.
