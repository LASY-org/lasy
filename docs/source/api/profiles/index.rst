Laser Profiles
==============
Laser pulses are incorporated into `lasy` via the `Profile` class.
Typically a laser will be constructed through a combination of two classes `LongitudinalProfile` and `TransverseProfile` which represent the longitudinal and transverse profiles  of the laser.

.. autoclass:: lasy.profiles.profile.Profile
    :members:


.. toctree::

   gaussian
   combined_profile
   longitudinal/index
   transverse/index