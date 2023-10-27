Overview of the Code
====================

``lasy`` manipulates laser pulses, and operates on the laser envelope.
It can be used to define the 3D profile of a laser pulse.
The user can define separately the transverse and longitudinal profile of the laser pulse either from a range common analytic profiles or using experimental measurements.
Once defined the laser pulse may be propagated to a user defined location.
Finally the laser profile may be outputted to file for use as an input to a variety of different simulation tools.

Codes Supporting LASY Data
##########################

A list of codes which support LASY Data as input:

* `FBPIC <https://github.com/fbpic/fbpic>`_
* `Wake-T <https://github.com/AngelFP/Wake-T>`_
* `WarpX <https://github.com/ECP-WarpX/WarpX>`_
* `HiPACE++ <https://github.com/Hi-PACE/hipace>`_

Structure of lasy
#################

All information pertaining to the representation of the laser pulse in the code is stored in the :doc:`laser <../api/laser>` object.
This contains both the physical and computational parameters.

The physical laser pulse parameters are defined in the laser :doc:`profile <../api/profiles/index>`.
This is typically constructed from a :doc:`combination <../api/profiles/combined_profile>` of two classes representing the :doc:`longitudinal <../api/profiles/longitudinal/index>` and :doc:`transverse <../api/profiles/transverse/index>` profiles of the laser.
Alternatively, one can define the full 3D profile in a single function, for example the :doc:`GaussianProfile <../api/profiles/gaussian>`

The data associated with a given laser pulse is stored on a grid. To create this grid and populate it with a laser pulse, we need to know something about the computational parameters being used.
For example, the metadata associated with this grid such as the coordinate system being used, lower and higher ends of the computational domain and number of points etc.
All of this information is passed when initializing the :doc:`laser <../api/laser>` object.

Once a laser :doc:`laser <../api/laser>` object has been defined, we can then propagate it forwards and backwards to see how it evolves or to set it in the right place for the beginning of a subsequent simulation.
The laser object can be written to a standard file format for these subsequent calculations. This allows for easy incorporation of standardized laser pulses to a range of different simulation tools.

Coordinate Systems
##################

In 3D (x,y,t) Cartesian coordinates, the definition used is:

.. math::
   \begin{aligned}
   E_x(x,y,t) = \operatorname{Re} \left( \mathcal{E}(x,y,t) e^{-i \omega_0t}p_x \right)\\
   E_y(x,y,t) = \operatorname{Re} \left( \mathcal{E}(x,y,t) e^{-i \omega_0t}p_y \right)\end{aligned}


where :math:`\operatorname{Re}` stands for real part,  :math:`E_x` (resp. :math:`E_y`) is the laser electric field in the :math:`x` (resp. :math:`y`) direction, :math:`\mathcal{E}` is the complex laser envelope stored and used in lasy, :math:`\omega_0 = 2\pi c/\lambda_0` is the angular frequency defined from the laser wavelength :math:`\lambda_0` and :math:`(p_x,p_y)` is the (complex and normalized) polarization vector.

In cylindrical coordinates, the envelope is decomposed in :math:`N_m` azimuthal modes ( see Ref. [A. Lifschitz et al., J. Comp. Phys. 228.5: 1803-1814 (2009)]). Each mode is stored on a 2D grid :math:`(r,t)`, using the following definition:

.. math::
   \begin{aligned}
   E_x (r,\theta,t) = \operatorname{Re}\left( \sum_{-N_m+1}^{N_m-1}\mathcal{E}_m(r,t) e^{-im\theta}e^{-i\omega_0t}p_x\right)\\
   E_y (r,\theta,t) = \operatorname{Re}\left( \sum_{-N_m+1}^{N_m-1}\mathcal{E}_m(r,t) e^{-im\theta}e^{-i\omega_0t}p_y\right).\end{aligned}
