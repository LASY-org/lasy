import numpy as np

from lasy.profiles.transverse.hermite_gaussian_profile import (
    HermiteGaussianTransverseProfile,
)
from lasy.profiles.transverse.laguerre_gaussian_profile import (
    LaguerreGaussianTransverseProfile,
)


def getHGMode(grid_in, w0x, w0y, i, j):
    r"""
    Function to project a laser field onto a Hermite-Gaussian mode to
    obtain the complex mode coefficient

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


def decomposeHG(grid_in, w0x, w0y, Mmax, Nmax, skipAsymmetricModes=False):
    r"""
    Function to decompose a laser field onto a Hermite-Gaussian basis

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
    X, Y, T = np.meshgrid(
        grid_in.grid.axes[0], grid_in.grid.axes[1], grid_in.grid.axes[2], indexing="ij"
    )
    field = grid_in.grid.get_temporal_field()

    cxy = {}
    for i in range(Mmax):
        for j in range(Nmax):
            if (i != j) and (skipAsymmetricModes):
                cxy[(i, j)] = 0
                continue
            cxy[(i, j)] = getHGMode(grid_in, w0x, w0y, i, j)

    return cxy


def reconstructHG(grid_in, w0x, w0y, cxy, skipAsymmetricModes=False):
    r"""
    Function to compose a laser field from a dictionary of complex
    modal coefficients for a Hermite-Gaussian basis and update the laser object

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


def getLGMode(grid_in, w0, i, j):
    r"""
    Function to project a laser field onto a Laguerre-Gaussian mode to
    obtain the complex mode coefficient

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


def decomposeLG(grid_in, w0, Mmax, Nmax, skipAsymmetricModes=False):
    r"""
    Function to decompose a laser field onto a Laguerre-Gaussian basis

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
    X, Y, T = np.meshgrid(
        grid_in.grid.axes[0], grid_in.grid.axes[1], grid_in.grid.axes[2], indexing="ij"
    )
    field = grid_in.grid.get_temporal_field()

    cxy = {}
    for i in range(Mmax):
        for j in range(Nmax):
            if (i != j) and (skipAsymmetricModes):
                cxy[(i, j)] = 0
                continue
            cxy[(i, j)] = getLGMode(grid_in, w0, i, j)

    return cxy


def reconstructLG(grid_in, w0, cxy, skipAsymmetricModes=False):
    r"""
    Function to compose a laser field from a dictionary of complex
    modal coefficients for a Laguerre-Gaussian basis and update the laser object

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
