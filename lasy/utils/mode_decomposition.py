import math

import numpy as np

from lasy.profiles.transverse.hermite_gaussian_profile import (
    HermiteGaussianTransverseProfile,
)
from lasy.profiles.transverse.transverse_profile import TransverseProfile
from lasy.utils.exp_data_utils import find_d4sigma


def hermite_gauss_decomposition(
    laserProfile,
    wavelength,
    res,
    lo,
    hi,
    m_max=12,
    n_max=12,
):
    """
    Decomposes a laser profile into a set of hermite-gaussian modes.

    The function takes an instance of `TransverseProfile`.

    Parameters
    ----------
    laserProfile : class instance
        An instance of a class or sub-class of TransverseLaserProfile

    wavelength : float (in meter)
        Central wavelength at which the Hermite-Gauss beams are to be defined.

    res : float
        The resolution of grid points in x and y that will be used
        during the decomposition calculation

    lo, hi : array of floats
        The lower and upper bounds of the spatial grid on which the
        decomposition will be performed.

    m_max, n_max : ints
        The maximum values of `m` and `n` up to which the expansion
        will be performed

    Returns
    -------
    weights : dict of floats
        A dictionary of floats corresponding to the weights of each mode
        in the decomposition. The keys of the dictionary are tuples
        corresponding to (`m`,`n`)

    waist : Beam waist for which the decomposition is calculated.
        It is computed as the waist for which the weight of order 0 is maximum.
    """
    # Check if the provided laserProfile is a transverse profile.
    assert isinstance(laserProfile, TransverseProfile), (
        "laserProfile must be an instance of TransverseProfile"
    )

    # Get the field, sensible spatial bounds for the profile
    lo0 = lo[0] + laserProfile.x_offset
    lo1 = lo[1] + laserProfile.x_offset
    hi0 = hi[0] + laserProfile.y_offset
    hi1 = hi[1] + laserProfile.y_offset

    Nx = int((hi0 - lo0) // (2 * res) * 2) + 2
    Ny = int((hi1 - lo1) // (2 * res) * 2) + 2

    # Define spatial arrays
    x = np.linspace(
        (lo0 + hi0) / 2 - (Nx - 1) / 2 * res,
        (lo0 + hi0) / 2 + (Nx - 1) / 2 * res,
        Nx,
    )
    y = np.linspace(
        (lo1 + hi1) / 2 - (Ny - 1) / 2 * res,
        (lo1 + hi1) / 2 + (Ny - 1) / 2 * res,
        Ny,
    )
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Get the field on this grid
    field = laserProfile.evaluate(X, Y)

    # Get estimate of w0
    w0 = estimate_best_HG_waist(x, y, field, wavelength)

    # Next we loop over the modes and calculate the relevant weights
    weights = {}
    for m in range(m_max):
        for n in range(n_max):
            HGMode = HermiteGaussianTransverseProfile(w0, w0, m, n, wavelength)
            coef = np.real(
                np.sum(field * HGMode.evaluate(X, Y)) * dx * dy
            )  # modalDecomposition
            if math.isnan(coef):
                coef = 0
            weights[(m, n)] = coef

    return weights, w0


def estimate_best_HG_waist(x, y, field, wavelength):
    """
    Estimate the waist that maximises the weighting of the first mode.

    Calculates a D4Sigma waist as a first estimate and then tests multiple
    gaussians with waists around this value to determine which has the best
    overlap with the provided intensity profile. The aim here is to maximise
    the energy in the fundamental mode of the reconstruction and so to avoid
    a decomposition with significant higher-order modal content.

    Parameters
    ----------
    x,y : 1D numpy arrays
        representing the x and y axes on which the intensity profile is defined.

    field : 2D numpy array representing the field (not the laser intensity).
        the laser field profile in a 2D slice.

    wavelength : float (in meter)
        Central wavelength at which the Hermite-Gauss beams are to be defined.

    Returns
    -------
    w0 : scalar
        The calculated waist.
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    assert np.isclose(dx, dy, rtol=1e-10)

    X, Y = np.meshgrid(x, y)

    D4SigX, D4SigY = find_d4sigma(np.abs(field) ** 2)
    w0Est = np.mean((D4SigX, D4SigY)) / 2 * dx  # convert this to a 1/e^2 width

    # Scan around the waist obtained from the D4sigma calculation,
    # and keep the waist for which this HG mode has the highest scalar
    # product with the input profile.
    waistTest = np.linspace(w0Est / 2, w0Est * 1.5, 30)
    coeffTest = np.zeros_like(waistTest)

    for i, wTest in enumerate(waistTest):
        # create a gaussian
        HGMode = HermiteGaussianTransverseProfile(wTest, wTest, 0, 0, wavelength)
        profile = HGMode.evaluate(X, Y)
        coeffTest[i] = np.real(np.sum(profile * field))
    w0 = waistTest[np.argmax(coeffTest)]

    print("Estimated w0 = %.2f microns (1/e^2 width)" % (w0Est * 1e6))
    return w0
