Laguerre Gaussian Transverse Profile
====================================

Used to define a Laguerre-Gaussian transverse laser profile.
Laguerre-Gaussian modes are a family of solutions to the paraxial wave equation written in cylindrical coordinates. The modes are characterised by a radial index :math:`p` and an azimuthal index :math:`m`.

The modes can have azimuthally varying fields (for :math:`m > 0`) but any single mode will always have an azimuthally invariant intensity profile.

.. image:: https://user-images.githubusercontent.com/27694869/211018326-d0ec3684-4ad9-4ed5-a252-18e2641ad402.png
    :alt: Laguerre-Gaussian-Modes

Laguerre-Gaussian beams can be added and scaled to create compositions of Laguerre-Gaussian beams::

    p1 = LaguerreGaussianTransverseProfile(20e-6, p=0, m=1)
    p2 = LaguerreGaussianTransverseProfile(20e-6, p=0, m=1)
    composition = 3 * p1 + p2 * 3

------------

.. autoclass:: lasy.profiles.transverse.laguerre_gaussian_profile.LaguerreGaussianTransverseProfile
    :members:
