Laser Profiles
==============

Laser pulses are incorporated into *lasy* via the ``Profile`` class.
Typically a laser will be constructed through a combination of two classes ``LongitudinalProfile`` and ``TransverseProfile`` which represent the longitudinal and transverse profiles  of the laser.

.. autoclass:: lasy.profiles.profile.Profile
    :members:

.. toctree::
   :hidden:

   gaussian
   combined_profile
   from_array_profile
   longitudinal/index
   transverse/index
