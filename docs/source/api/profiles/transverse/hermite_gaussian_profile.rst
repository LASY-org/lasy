Hermite Gaussian Transverse Profile
===================================

Used to define a Hermite-Gaussian transverse laser profile.
Hermite-Gaussian modes are a family of solutions to the paraxial wave equation written in cartesian coordinates. The modes are characterised by two transverse indices :math:`n_x` and :math:`n_y`.

.. image:: https://user-images.githubusercontent.com/27694869/211018259-15d925bb-f123-42c6-b5ba-86488356ae70.png
    :alt: Hermite-Gauss-Modes

Hermite Gaussian beams can be added and scaled to create compositions of Hermite-Gaussian beams::
    
    p1 = HermiteGaussianTransverseProfile(20e-6, n_x=2, n_y=1)
    p2 = HermiteGaussianTransverseProfile(20e-6, n_x=0, n_y=1)
    composition = 3 * p1 + p2 * 3


------------

.. autoclass:: lasy.profiles.transverse.hermite_gaussian_profile.HermiteGaussianTransverseProfile
    :members:
