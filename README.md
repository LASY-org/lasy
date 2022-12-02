# lasy

## Documentation

LASY manipulates laser pulses, and operates on the laser envelope. The definition used is:

```math
   \begin{aligned}
   E_x(x,y,t) = \Re ( \mathcal{E}(x,y,t) e^{-i\omega_0t}p_x)\\
   E_y(x,y,t) = \Re ( \mathcal{E}(x,y,t) e^{-i\omega_0t}p_y)\end{aligned}
```

where $E_x$ (resp. $E_y$) is the laser electric field in the x (resp. y) direction, $\mathcal{E}$ is the complex laser envelope stored and used in lasy, $\omega_0 = 2\pi c/\lambda_0$ is the angular frequency defined from the laser wavelength $\lambda_0$ and $(p_x,p_y)$ is the (complex and normalized) polarization vector.
 
## Style conventions

- Docstrings are written using the Numpy style.
- For each significant contribution, using pull requests is encouraged: the description helps to explain the code and open dicussion.

## Test

```bash
python setup.py install
python examples/test.py
```
