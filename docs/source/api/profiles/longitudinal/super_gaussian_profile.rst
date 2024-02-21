Super-Gaussian Longitudinal Profile
===================================

Used to define a super-Gaussian transverse laser profile.

The shape of the profile is characterised by the duration :math:`\tau` and by one "order parameter" :math:`n`, where :math:`n=2` gives a standard Gaussian profile, and the profile converges to a square pulse when :math:`n` goes to infinity.

The FWHM (amplitude of the field) of the profile (:math:`t_{fwhm}`) is related to :math:`\tau` by :math:`\tau = \frac{1}{2}\ln(2)^{-\frac{1}{n}}`

.. autoclass:: lasy.profiles.longitudinal.SuperGaussianLongitudinalProfile
    :members:
