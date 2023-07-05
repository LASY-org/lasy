import numpy as np
import math


def get_zernike_nm(j):
    """
    Convert between different Zernike index schemes.

    Convert the OSA/ANSI Zernike Polynomial Index to the
    standard n,m indexing

    Parameters
    ----------
    j : int
        The OSA/ANSI Zernike Index

    Returns
    -------
    n,m : ints
        The standard Zernike Polynomial Indexes
    """
    n = int(np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2))
    m = 2 * j - n * (n + 2)
    return int(m), int(n)


def zernike(x, y, pupilCoords, j):
    """
    Calculate the Zernike Polynomials to arbitrary order.

    Makes use of constructor formula on https://en.wikipedia.org/wiki/Zernike_polynomials

    Parameters
    ----------
    x, y : ndarrays (meters)
        The position at which to calculate the profile

    pupilCoords : tuple of floats (meters)
        A tuple of floats (cgx,cgy,r) with the first two elements corresponding to the center
        of the zernike mode and the third the radius of the mode

    j : int
        The OSA/ANSI Zernike Index

    Returns
    -------
    Z : ndarray (rad)
        The Zernike mode
    """
    # Setup
    (cgx, cgy, r) = pupilCoords
    rho = np.sqrt((x - cgx) ** 2 + (y - cgy) ** 2) / r
    theta = np.arctan2(y - cgy, x - cgx)

    m, n = get_zernike_nm(j)

    # next get the radial part
    R = RmnGenerator(n, abs(m), rho)

    # Now multiply by the azimuthal part
    if m < 0:
        Z = R * np.sin(-m * theta)
    else:
        Z = R * np.cos(m * theta)

    # Normalization
    if n == 0:
        scaling = 1
    else:
        if m == 0:
            scaling = np.sqrt((n + 1))
        else:
            scaling = np.sqrt(2 * (n + 1))
    Z = Z * scaling

    return Z


def RmnGenerator(n, m, rho):
    """
    Generate the Radial component of the Zernike Polynomials.

    Parameters
    ----------
    n,m : ints
        The standard Zernike Polynomial Indices

    rho : ndarray (meters)
        The radial positions at which to calculate the profile

    Returns
    -------
    Rmn : ndarray (rad)
        The radial component of the Zernike mode
    """
    if n == 0:
        try:
            (r,) = rho.shape
            Rmn = np.ones(
                r,
            )
        except:
            r, c = rho.shape
            Rmn = np.ones((r, c))
    elif (n - m) % 2 == 0:
        # Even, Rmn is not 0
        k = np.linspace(0, int((n - m) / 2), int((n - m) / 2) + 1).astype(int)
        try:
            (r,) = rho.shape
            Rmn = np.zeros(
                r,
            )
        except:
            r, c = rho.shape
            Rmn = np.zeros((r, c))
        for i in k:
            Rmn = Rmn + ((-1) ** i * math.factorial(n - i)) / (
                math.factorial(i)
                * math.factorial(int((n + m) / 2) - i)
                * math.factorial(int((n - m) / 2) - i)
            ) * rho ** (n - 2 * i)

    else:
        try:
            (r,) = rho.shape
            Rmn = np.zeros(
                r,
            )
        except:
            r, c = rho.shape
            Rmn = np.zeros((r, c))

    return Rmn
