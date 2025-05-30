# lasy
[![Documentation Status](https://readthedocs.org/projects/LASY/badge/?version=latest)](https://lasydoc.readthedocs.io/en/latest/)
![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Paaaaarth/LASY/Badges/coverage-badge.json)
[![PyPI](https://img.shields.io/pypi/v/LASY)](https://pypi.org/project/lasy/)
[![License](https://img.shields.io/badge/License-view-blue)](https://github.com/LASY-org/lasy/blob/development/license.txt)

## Overview

`lasy` is a Python library that facilitates the initialization of complex laser pulses, in simulations of laser-plasma interactions.

More specifically, `lasy` offers many ways to define complex laser pulses (e.g. from commonly-known analytical formulas, from experimental measurements, etc.) and offers pre-processing functionalities (e.g. propagation, re-normalization, geometry conversion). The laser field is then exported in a standardized file, that can be read by external simulation codes.

## Installation

For the standard release of the code simply run:
```
python3 -m pip install lasy
```

If you would prefer the most recent version of the code with the latest functionalities, then run:
```
python3 -m pip install git+https://github.com/LASY-org/lasy.git
```

## Tutorials
An interactive Google Colab tutorial showing some of the main functionalities of lasy can be found [here](https://colab.research.google.com/drive/1nPwgIUea6Jhzc9CSPDZXCcviWCmebjF1?usp=sharing).

This document is updated on a best-effort basis to keep track with developments in the lasy code base.

Additionally, a set of static (automatically tested) examples can be found [here](https://lasydoc.readthedocs.io/en/latest/tutorials/index.html).


## Documentation

LASY manipulates laser pulses, and operates on the laser envelope. In 3D (x,y,t) Cartesian coordinates, the definition used is:

```math
   \begin{aligned}
   E_x(x,y,t) = \mathrm{Re}\left( \mathcal{E}(x,y,t) e^{-i\omega_0t}p_x\right)\\
   E_y(x,y,t) = \mathrm{Re}\left( \mathcal{E}(x,y,t) e^{-i\omega_0t}p_y\right)\end{aligned}
```

where $\mathrm{Re}$ stands for real part,  $E_x$ (resp. $E_y$) is the laser electric field in the x (resp. y) direction, $\mathcal{E}$ is the complex laser envelope stored and used in lasy, $\omega_0 = 2\pi c/\lambda_0$ is the angular frequency defined from the laser wavelength $\lambda_0$ and $(p_x,p_y)$ is the (complex and normalized) polarization vector.

In cylindrical coordinates, the envelope is decomposed in $N_m$ azimuthal modes ( see Ref. [A. Lifschitz et al., J. Comp. Phys. 228.5: 1803-1814 (2009)]). Each mode is stored on a 2D grid (r,t), using the following definition:

```math
   \begin{aligned}
   E_x (r,\theta,t) = \mathrm{Re}\left( \sum_{-N_m+1}^{N_m-1}\mathcal{E}_m(r,t) e^{-im\theta}e^{-i\omega_0t}p_x\right)\\
   E_y (r,\theta,t) = \mathrm{Re}\left( \sum_{-N_m+1}^{N_m-1}\mathcal{E}_m(r,t) e^{-im\theta}e^{-i\omega_0t}p_y\right).\end{aligned}
```

For more information, please check our [arXiv preprint](https://doi.org/10.48550/arXiv.2403.12191).

## Workflow

# How to contribute

All contributions are welcome! For a new contribution, we use pull requests from forks. Below is a very rough summary, please have a look at the appropriate documentation at https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks and around.

First, setup your fork workflow (only once):
- Fork the repo by clicking the Fork button on the top right, and follow the prompts. This will create your own (remote) copy of the main https://github.com/LASY-org/LASY repo, located at https://github.com/[yourusername]/LASY.
- Make your local copy aware of your fork: from your local repository, do `git remote add [some-name] https://github.com/[your username]/LASY`. For `[some-name]` it can be convenient to use e.g. your username.

Then, for each contribution:
- Get the last version of branch `development` from the main repo (e.g. `git checkout development && git pull`).
- Create a new branch (e.g. `git checkout -b my_contribution`).
- Do usual `git add` and `git commit` operations.
- Push your branch to your own fork: `git push -u [some-name] my_contribution`
- Whenever you're ready, open a PR from branch `my_contribution` on your fork to branch `development` on the main repo. Github typically suggests this very well.

# Style conventions

- Docstrings are written using the Numpy style.
- Functions in `utils/laser_utils.py` only depend on standard types (Python & Numpy) and on the `Grid` class. That way, they are relatively stand-alone and can be used on different data structures. A simple Grid factory is provided for that purpose.
- A PR should be open for any contribution: the description helps to explain the code and open dicussion.



## Testing

For tests, you need to have a few extra packages, such as `pytest` and `openpmd-viewer` installed:
```bash
python3 -m pip install -r tests/requirements.txt
```

After successful installation, you can run the unit tests:
```bash
# Run all tests
python3 -m pytest tests/

# Run tests from a single file
python3 -m pytest tests/test_laser_profiles.py

# Run a single test (useful during debugging)
python3 -m pytest tests/test_laser_profiles.py::test_profile_gaussian_3d_cartesian

# Run all tests, do not capture "print" output and be verbose
python3 -m pytest -s -vvvv tests/
```
## Creating Documentation

Install sphinx (https://www.sphinx-doc.org/en/master/usage/installation.html)

```bash
python -m pip install --upgrade -r docs/requirements.txt
cd docs
sphinx-build -b html source _build
```
