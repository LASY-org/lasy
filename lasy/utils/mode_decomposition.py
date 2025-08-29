import numpy as np

from lasy.profiles.transverse.hermite_gaussian_profile import (
    HermiteGaussianTransverseProfile,
)
from lasy.profiles.transverse.laguerre_gaussian_profile import (
    LaguerreGaussianTransverseProfile,
)
from lasy.utils.exp_data_utils import find_d4sigma


def get_hermite_mode(grid_in, w0x, w0y, i, j):
    r"""
    Project a laser field onto a Hermite-Gaussian mode to
    obtain the complex mode coefficient.

    Parameters
    ----------
    grid_in : Grid
        Grid object at the input plane.

    w0x : float (in m)
        Spot size in the x-direction

    w0y : float (in m)
        Spot size in the y-direction

    i : integer
        Order of the x-direction mode

    j : integer
        Order of the y-direction mode

    Returns
    -------
    coeff : complex float
        The projected complex modal coefficient for the (i,j) Hermite-Gaussian mode
    """
    X, Y, T = np.meshgrid(
        grid_in.grid.axes[0], grid_in.grid.axes[1], grid_in.grid.axes[2], indexing="ij"
    )

    dx = np.mean(np.diff(grid_in.grid.axes[0]))
    dy = np.mean(np.diff(grid_in.grid.axes[1]))

    hg = HermiteGaussianTransverseProfile(
        w0x, w0y, i, j, grid_in.profile.lambda0
    ).evaluate(X, Y)

    coeff = np.sum(grid_in.grid.get_temporal_field() * np.conj(hg)) * dx * dy

    return coeff


def hermite_gauss_decomposition(
    grid_in, w0x, w0y, Mmax, Nmax, skipAsymmetricModes=False
):
    r"""
    Decompose a laser field onto a Hermite-Gaussian basis.

    Parameters
    ----------
    grid_in : Grid
        Grid object at the input plane.

    w0x : float (in m)
        Spot size in the x-direction

    w0y : float (in m)
        Spot size in the y-direction

    Mmax : integer
        Maximum order of the x-direction mode

    Nmax : integer
        Maximum order of the y-direction mode

    skipAsymmetricModes : Boolean
        Allows the user to only consider symmetric modal coefficients

    Returns
    -------
    cxy : dict
        A dictionary of complex modal coefficients
    """
    cxy = {}
    for i in range(Mmax):
        for j in range(Nmax):
            if (i != j) and (skipAsymmetricModes):
                cxy[(i, j)] = 0
                continue
            cxy[(i, j)] = get_hermite_mode(grid_in, w0x, w0y, i, j)

    return cxy


def hermite_gauss_composition(grid_in, w0x, w0y, cxy, skipAsymmetricModes=False):
    r"""
    Compose a laser field from a dictionary of complex
    modal coefficients for a Hermite-Gaussian basis and update the laser object.

    Parameters
    ----------
    grid_in : Grid
        Grid object at the input plane.

    w0x : float (in m)
        Spot size in the x-direction

    w0y : float (in m)
        Spot size in the y-direction

    cxy : dict
        A dictionary of complex modal coefficients

    skipAsymmetricModes : Boolean
        Allows the user to only consider symmetric modal coefficients
    """
    X, Y, T = np.meshgrid(
        grid_in.grid.axes[0], grid_in.grid.axes[1], grid_in.grid.axes[2], indexing="ij"
    )
    field = np.zeros(X.shape, dtype="complex128")

    for nxy in list(cxy):
        if (nxy[0] != nxy[1]) and (skipAsymmetricModes):
            continue
        field += cxy[nxy] * (
            HermiteGaussianTransverseProfile(
                w0x, w0y, nxy[0], nxy[1], grid_in.profile.lambda0
            ).evaluate(X, Y)
        )

    grid_in.grid.set_temporal_field(field)


def get_laguerre_mode(grid_in, w0, i, j):
    r"""
    Project a laser field onto a Laguerre-Gaussian mode to
    obtain the complex mode coefficient.

    Parameters
    ----------
    grid_in : Grid
        Grid object at the input plane.

    w0 : float (in m)
        Spot size

    i : integer
        Order of the radial mode

    j : integer
        Order of the azimuthal mode

    Returns
    -------
    coeff : complex float
        The projected complex modal coefficient for the (i,j) Laguerre-Gaussian mode
    """
    X, Y, T = np.meshgrid(
        grid_in.grid.axes[0], grid_in.grid.axes[1], grid_in.grid.axes[2], indexing="ij"
    )

    dx = np.mean(np.diff(grid_in.grid.axes[0]))
    dy = np.mean(np.diff(grid_in.grid.axes[1]))

    lg = LaguerreGaussianTransverseProfile(w0, i, j, grid_in.profile.lambda0).evaluate(
        X, Y
    )

    coeff = np.sum(grid_in.grid.get_temporal_field() * np.conj(lg)) * dx * dy

    return coeff


def laguerre_gauss_decomposition(grid_in, w0, Mmax, Nmax, skipAsymmetricModes=False):
    r"""
    Decompose a laser field onto a Laguerre-Gaussian basis.

    Parameters
    ----------
    grid_in : Grid
        Grid object at the input plane.

    w0 : float (in m)
        Spot size

    Mmax : integer
        Maximum order of the radial mode

    Nmax : integer
        Maximum order of the azimuthal mode

    skipAsymmetricModes : Boolean
        Allows the user to only consider symmetric modal coefficients

    Returns
    -------
    cxy : dict
        A dictionary of complex modal coefficients
    """
    cxy = {}
    for i in range(Mmax):
        for j in range(Nmax):
            if (i != j) and (skipAsymmetricModes):
                cxy[(i, j)] = 0
                continue
            cxy[(i, j)] = get_laguerre_mode(grid_in, w0, i, j)

    return cxy


def laguerre_gauss_composition(grid_in, w0, cxy, skipAsymmetricModes=False):
    r"""
    Compose a laser field from a dictionary of complex
    modal coefficients for a Laguerre-Gaussian basis and update the laser object.

    Parameters
    ----------
    grid_in : Grid
        Grid object at the input plane.

    w0 : float (in m)
        Spot size

    cxy : dict
        A dictionary of complex modal coefficients

    skipAsymmetricModes : Boolean
        Allows the user to only consider symmetric modal coefficients
    """
    X, Y, T = np.meshgrid(
        grid_in.grid.axes[0], grid_in.grid.axes[1], grid_in.grid.axes[2], indexing="ij"
    )
    field = np.zeros(X.shape, dtype="complex128")

    for nxy in list(cxy):
        if (nxy[0] != nxy[1]) and (skipAsymmetricModes):
            continue
        field += cxy[nxy] * (
            LaguerreGaussianTransverseProfile(
                w0, nxy[0], nxy[1], grid_in.profile.lambda0
            ).evaluate(X, Y)
        )

    grid_in.grid.set_temporal_field(field)


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
    w0x, w0y : floats
        The calculated waist in x and y axis.
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    assert np.isclose(dx, dy, rtol=1e-10)

    X, Y = np.meshgrid(x, y)

    D4SigX, D4SigY = find_d4sigma(np.abs(field) ** 2)
    # convert this to a 1/e^2 width
    w0EstX = np.mean(D4SigX) / 2 * dx
    w0EstY = np.mean(D4SigY) / 2 * dy

    # Scan around the waist obtained from the D4sigma calculation,
    # and keep the waist for which this HG mode has the highest scalar
    # product with the input profile.
    waistTestX = np.linspace(w0EstX / 2, w0EstX * 1.5, 30)
    waistTestY = np.linspace(w0EstY / 2, w0EstY * 1.5, 30)
    coeffTest = np.zeros_like(waistTestX)

    for i in range(30):
        # create a gaussian
        HGMode = HermiteGaussianTransverseProfile(
            waistTestX[i], waistTestY[i], 0, 0, wavelength
        )
        profile = HGMode.evaluate(X, Y)
        coeffTest[i] = np.real(np.sum(profile * field))
    w0x = waistTestX[np.argmax(coeffTest)]
    w0y = waistTestY[np.argmax(coeffTest)]

    print("Estimated w0(x-axis) = %.2f microns (1/e^2 width)" % (w0x * 1e6))
    print("Estimated w0(y-axis) = %.2f microns (1/e^2 width)" % (w0y * 1e6))
    return w0x, w0y
