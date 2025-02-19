from lasy.profiles.transverse.hermite_gaussian_profile import (
    HermiteGaussianTransverseProfile,
)
from lasy.profiles.transverse.transverse_profile import TransverseProfile
from lasy.utils.mode_decomposition import hermite_gauss_decomposition


def hg_reconstruction(
    transverse_profile,
    wavelength,
    resolution,
    lo,
    hi,
    n_modes_x=10,
    n_modes_y=10,
):
    """
    Denoise the transverse profile by decomposing it into a set of Hermite-Gaussian modes.

    The profiles are weighted according to mode coefficients and then added.

    Parameters
    ----------
    transverse_profile : TransverseProfile object
        An instance of a class or sub-class of TransverseProfile.
        Defines the transverse envelope of the laser.

    wavelength : float (in meter)
        Central wavelength at which the Hermite-Gauss beams are to be defined.

    resolution : float
        The number of grid points in x and y used during the decomposition calculation.

    lo, hi : array of floats
        The lower and upper bounds of the spatial grid on which the
        decomposition is be performed.

    n_modes_x, n_modes_y : ints (optional)
        The maximum values of `n_x` and `n_y` up to which the expansion is performed.

    Returns
    -------
    transverse_profile_cleaned : class instance
        Denoised transverse profile after decomposition and recombination.

    w0x, w0y : floats
        Beam waist for which the decomposition is calculated.
        It is computed as the waist for which the weight of order 0 is maximum.
    """
    assert isinstance(transverse_profile, TransverseProfile)

    # Calculate the decomposition and waist of the laser pulse
    modeCoeffs, w0x, w0y = hermite_gauss_decomposition(
        transverse_profile, wavelength, resolution, lo, hi, n_modes_x, n_modes_y
    )

    # Denosing the laser profile
    for i, mode_key in enumerate(list(modeCoeffs)):
        transverse_profile_temp = HermiteGaussianTransverseProfile(
            w0x, w0y, mode_key[0], mode_key[1], wavelength
        )  # Create a new profile for each mode

        if i == 0:  # First mode (0,0)
            transverse_profile_cleaned = modeCoeffs[mode_key] * transverse_profile_temp
        else:  # All other modes
            transverse_profile_cleaned += modeCoeffs[mode_key] * transverse_profile_temp

    return transverse_profile_cleaned, w0x, w0y
