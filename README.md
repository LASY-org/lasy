# lasy

## Documentation

LASY manipulates laser pulses, and operates on the laser envelope. In 3D (x,y,t) Cartesian coordinates, the definition used is:

```math
   \begin{aligned} 
   E_x(x,y,t) = \operatorname{Re}\left( \mathcal{E}(x,y,t) e^{-i\omega_0t}p_x\right)\\
   E_y(x,y,t) = \operatorname{Re}\left( \mathcal{E}(x,y,t) e^{-i\omega_0t}p_y\right)\end{aligned}
```

where $\operatorname{Re}$ stands for real part,  $E_x$ (resp. $E_y$) is the laser electric field in the x (resp. y) direction, $\mathcal{E}$ is the complex laser envelope stored and used in lasy, $\omega_0 = 2\pi c/\lambda_0$ is the angular frequency defined from the laser wavelength $\lambda_0$ and $(p_x,p_y)$ is the (complex and normalized) polarization vector.

In cylindrical coordinates, the envelope is decomposed in $N_m$ azimuthal modes ( see Ref. [A. Lifschitz et al., J. Comp. Phys. 228.5: 1803-1814 (2009)]). Each mode is stored on a 2D grid (r,t), using the following definition:

```math
   \begin{aligned}
   E_x (r,\theta,t) = \operatorname{Re}\left( \sum_{-N_m+1}^{N_m-1}\mathcal{E}_m(r,t) e^{-im\theta}e^{-i\omega_0t}p_x\right)\\
   E_y (r,\theta,t) = \operatorname{Re}\left( \sum_{-N_m+1}^{N_m-1}\mathcal{E}_m(r,t) e^{-im\theta}e^{-i\omega_0t}p_y\right).\end{aligned}
```

At the moment LASY only support axisymmetric envelope profiles: $N_m=1$.
 
## Workflow

# How to contribute

All contributions are welcome! For all contribution, we use pull requests from forks. Below is a very rough summary, please have a look at the appropriate documentation at https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks and around.

First, setup your fork workflow (only once):
- Fork the repo by clicking the Fork button on the top right, and follow the prompts. This will create your own (remote) copy of the main https://github.com/LASY-org/LASY repo, located at https://github.com/<your username>/LASY.
- Make your local copy aware of your fork: from your local repository, do `git remote add <blah> https://github.com/<your username>/LASY`. For `<blah>` it can be convenient to use e.g. your username.

Then, for each contribution:
- Get the last version of branch `development` from the main repo (e.g. `git checkout development && git pull`).
- Create a new branch (e.g. `git checkout -b my_contribution`).
- Do usual `git add` and `git commit` operations.
- Push your branch to your own fork: `git push -u <blah> my_contribution`
- Whenever you're ready, open a PR from branch `my_contribution` on your fork to branch `development` on the main repo. Github typically suggests this very well.

# Style conventions

- Docstrings are written using the Numpy style.
- A PR should be open for any contribution: the description helps to explain the code and open dicussion.

## Test

```bash
python setup.py install
python examples/test.py
```
