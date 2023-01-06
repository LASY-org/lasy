Laguerre Gaussian Transverse Profile
====================================

Used to define a Laguerre-Gaussian transverse laser profile. 
Laguerre-Gaussian modes are a family of solutions to the paraxial
wave equation written in cylindrical coordinates. The modes are
characterised by a radial index :math:`p` and an azimuthal index
:math:`m`. 

The modes can have azimuthally varying fields (for :math:`m > 0`)
but any single mode will always have an azimuthally invariant 
intensity profile.

.. image:: ../../../_static/laguerre_gauss_modes.png


------------

.. autoclass:: lasy.profiles.transverse.laguerre_gaussian_profile.LaguerreGaussianTransverseProfile
    :members: