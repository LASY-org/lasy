import numpy as np
from .speckle_profile import SpeckleProfile

class GPISIProfile(SpeckleProfile):
    """Generate a speckled laser profile with smoothing by spectral dispersion (SSD).

    This has temporal smoothing.

    Parameters
    ----------

    relative_laser_bandwidth : float
        Bandwidth :math:`\Delta_\nu` of the laser pulse, relative to central frequency.

    """
    def __init__(self, *speckle_args, 
        relative_laser_bandwidth,
    ):
        super().__init__(*speckle_args)
        self.laser_bandwidth = relative_laser_bandwidth
        
    def beamlets_complex_amplitude(
        self, t_now,
    ):
        """Calculate complex amplitude of the beamlets in the near-field, before propagating to the focal plane.

        Parameters
        ----------

        Returns
        -------
        array of complex numbers giving beamlet amplitude and phases in the near-field
        """

        return 
    