from lasy.profiles.transverse.transverse_profile import TransverseProfile
from lasy.profiles.transverse.transverse_profile_from_data import (
    TransverseProfileFromData,
)
from lasy.profiles.transverse.hermite_gaussian_profile import (
    HermiteGaussianTransverseProfile,
)
from lasy.utils.exp_data_utils import find_d4sigma

import numpy as np
import math


def hermite_gauss_decomposition(laserProfile, n_x_max=12, n_y_max=12, res=1e-6):
    """
    Decomposes a laser profile into a set of hermite-gaussian modes.

    The function takes either an instance of `TransverseProfile` or an
    instance of `Laser` (that is, either a transverse profile or the
    full 3D laser profile defined on a grid). In the case that an
    instance of `Laser` is passed then the intensity of this profile
    is projected onto an x-y plane for the decomposition.

    Parameters
    ----------
    laserProfile : class instance
        An instance of a class or sub-class of TransverseLaserProfile

    n_x_max, n_y_max : ints
        The maximum values of `n_x` and `n_y` out to which the expansion
        will be performed

    res : float
        The resolution of grid points in x and y that will be used
        during the decomposition calculation

    Returns
    -------
    weights : dict of floats
        A dictionary of floats corresponding to the weights of each mode
        in the decomposition. The keys of the dictionary are tuples
        corresponding to (`n_x`,`n_y`)

    waist : Beam waist for which the decomposition is calculated.
        It is computed as the waist for which the weight of order 0 is maximum.
    """
    # Check if the provided laserProfile is a full laser profile or a
    # transverse profile.

    assert isinstance(
        laserProfile, TransverseProfile
    ), "laserProfile must be an instance of TransverseProfile"

    # Get the field, sensible spatial bounds for the profile
    lo = [None, None]
    hi = [None, None]
    if isinstance(laserProfile, TransverseProfileFromData):
        lo[0] = laserProfile.field_interp.grid[0].min() + laserProfile.x_offset
        lo[1] = laserProfile.field_interp.grid[1].min() + laserProfile.x_offset
        hi[0] = laserProfile.field_interp.grid[0].max() + laserProfile.y_offset
        hi[1] = laserProfile.field_interp.grid[1].max() + laserProfile.y_offset

    else:
        lo[0] = -laserProfile.w0 * 5 + laserProfile.x_offset
        lo[1] = -laserProfile.w0 * 5 + laserProfile.x_offset
        hi[0] = laserProfile.w0 * 5 + laserProfile.x_offset
        hi[1] = laserProfile.w0 * 5 + laserProfile.x_offset

    N_pts_x = int((hi[0] - lo[0]) / res)
    N_pts_y = int((hi[1] - lo[1]) / res)

    # Define spatial arrays
    x = np.linspace(lo[0], hi[0], N_pts_x)
    y = np.linspace(lo[1], hi[1], N_pts_y)
    X, Y = np.meshgrid(x, y)
    dx = x[2] - x[1]
    dy = y[2] - y[1]

    # Get the field on this grid
    field = laserProfile.evaluate(X, Y)

    # Get estimate of w0
    w0 = estimate_best_HG_waist(x, y, field)

    # Next we loop over the modes and calculate the relevant weights
    weights = {}
    for i in range(n_x_max):
        for j in range(n_y_max):
            HGMode = HermiteGaussianTransverseProfile(w0, i, j)
            coef = np.sum(field * HGMode.evaluate(X, Y)) * dx * dy  # modalDecomposition
            if math.isnan(coef):
                coef = 0
            weights[(i, j)] = coef

    return weights, w0


def estimate_best_HG_waist(x, y, field):
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

    Returns
    -------
    w0 : scalar
        The calculated waist.
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    assert dx == dy

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
        HGMode = HermiteGaussianTransverseProfile(wTest, 0, 0)
        profile = HGMode.evaluate(X, Y)
        coeffTest[i] = np.sum(profile * field)
    w0 = waistTest[np.argmax(coeffTest)]

    print("Estimated w0 = %.2f microns" % (w0Est * 1e6))
    return w0
