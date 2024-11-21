from lasy.profiles.transverse.hermite_gaussian_profile import (
    HermiteGaussianTransverseProfile,
)
from lasy.utils.mode_decomposition import hermite_gauss_decomposition


def denoise_transverse_hg(
    transverse_profile, resolution=0.2e-6, n_modes_x=2, n_modes_y=2 , lo = [-2e-4,2e-4], hi = [2e-4,2e-4]
):
    """
    Denoise the transverse profile by decomposing it into a set of Hermite-Gaussian modes.

    The profiles are weighted according to mode coefficients and then added.

    Parameters
    ----------
    transverse_profile : class instance
        An instance of a class or sub-class of TransverseProfile.
        Defines the transverse envelope of the laser.

    resolution : float
        The resolution of grid points in x and y that will be used
        during the decomposition calculation.

    n_modes_x, n_modes_y : ints
        The maximum values of `n_x` and `n_y` out to which the
        expansion will be performed.

    Returns
    -------
    transverse_profile_cleaned : class instance
        Denoised transverse profile after decomposition and recombination.

    waist : float (meter)
        Beam waist for which the decomposition is calculated.
        It is computed as the waist for which the weight of order 0 is maximum.

    laser_energy_new : float
        The total energy of the laser pulse after decomposition.
    """
    laser_energy_new = 0

    # Calculate the decomposition and waist of the laser pulse
    modeCoeffs, waist = hermite_gauss_decomposition(
        transverse_profile, n_modes_x, n_modes_y, resolution ,lo, hi
    )

    # Denosing the laser profile
    for i, mode_key in enumerate(list(modeCoeffs)):
        transverse_profile_temp = HermiteGaussianTransverseProfile(
            waist, mode_key[0], mode_key[1]
        )  # Create a new profile for each mode

        print(f"Mode {i}: {mode_key} with coefficient {modeCoeffs[mode_key]}")
        laser_energy_new += modeCoeffs[mode_key] ** 2  # Energy fraction of the mode

        if i == 0:  # First mode (0,0)
            transverse_profile_cleaned = modeCoeffs[mode_key] * transverse_profile_temp
        else:  # All other modes
            transverse_profile_cleaned += modeCoeffs[mode_key] * transverse_profile_temp

    # Energy loss due to decomposition
    energy_loss = 1 - laser_energy_new
    print(f"Energy loss: {energy_loss * 100:.2f}%")

    return transverse_profile_cleaned, waist, laser_energy_new
