# lasy

## Documentation

LASY manipulates laser pulses, and operates on the laser envelope. In 3D (x,y,t) Cartesian coordinates, the definition used is:

```math
   \begin{aligned} 
   E_x(x,y,t) = \operatorname{Re}\left( \mathcal{E}(x,y,t) e^{-i\omega_0t}p_x\right)\\
   E_y(x,y,t) = \operatorname{Re}\left( \mathcal{E}(x,y,t) e^{-i\omega_0t}p_y\right)\end{aligned}
```

where $E_x$ (resp. $E_y$) is the laser electric field in the x (resp. y) direction, $\mathcal{E}$ is the complex laser envelope stored and used in lasy, $\omega_0 = 2\pi c/\lambda_0$ is the angular frequency defined from the laser wavelength $\lambda_0$ and $(p_x,p_y)$ is the (complex and normalized) polarization vector.

In cylindrical coordinates, the envelope is decomposed in $N_m$ azimuthal modes ( see Ref. [A. Lifschitz et al., J. Comp. Phys. 228.5: 1803-1814 (2009)]). Each mode is stored on a 2D grid (r,t), using the following definition:

```math
   \begin{aligned}
   E_x (r,\theta,t) = \operatorname{Re}\left( \sum_{-N_m+1}^{N_m-1}\mathcal{E}_m(r,t) e^{-im\theta}e^{-i\omega_0t}p_x\right)\\
   E_y (r,\theta,t) = \operatorname{Re}\left( \sum_{-N_m+1}^{N_m-1}\mathcal{E}_m(r,t) e^{-im\theta}e^{-i\omega_0t}p_y\right).\end{aligned}
```

At the moment LASY only support axisymmetric envelope profiles ($N_m=1$).
 
## Style conventions

- Docstrings are written using the Numpy style.
- For each significant contribution, using pull requests is encouraged: the description helps to explain the code and open dicussion.

## Test

```bash
python setup.py install
python examples/test.py
```
