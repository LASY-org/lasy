import numpy as np
from .speckle_profile import SpeckleProfile

class PhasePlateProfile(SpeckleProfile):
    """Generate a speckled laser profile with a random phase plate.

    This has no temporal smoothing.

    Parameters
    ----------
    rpp_cpp: string, keyword only, can be 'rpp' or 'cpp'
    """
    def __init__(self, *speckle_args, rpp_cpp):
        super().__init__(*speckle_args)
        self.rpp_cpp = rpp_cpp

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
        if self.rpp_cpp.upper() == "RPP":
            phase_plate = np.random.choice([0, np.pi], self.n_beamlets)
        else:
            phase_plate = np.random.uniform(
                -np.pi, np.pi, size=self.n_beamlets[0] * self.n_beamlets[1]
            ).reshape(self.n_beamlets)
        exp_phase_plate = np.exp(1j * phase_plate)
        return exp_phase_plate
    