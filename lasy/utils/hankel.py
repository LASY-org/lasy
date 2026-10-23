# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL

import numpy as np
from scipy.special import jn, jn_zeros

from lasy.backend import to_gpu, xp


class DHT(object):
    """Class to compute the discrete Hankel transform."""

    def __init__(self, m, Nr, Nz, rmax):
        """
        Initialize the Hankel transform.

        Parameters
        ----------
        m: int
        The azimuthal mode for which the Hankel transform is calculated

        Nr, Nz: float
        Number of points in the r direction and z direction

        rmax: float
        Edge of the box in which the Hankel transform is taken
        (The function is assumed to be zero at that point.)

        use_cuda: bool, optional
        Whether to use the GPU for the Hankel transform
        """
        # Register values of the arguments
        self.m = m
        self.Nr = Nr
        self.rmax = rmax
        self.Nz = Nz

        # Calculate the zeros of the Bessel function
        if m != 0:
            # In this case, 0 is a zero of the Bessel function of order m.
            # It turns out that it is needed to reconstruct the signal for p=0.
            alphas = np.hstack((np.array([0.0]), jn_zeros(m, Nr - 1)))
        else:
            alphas = jn_zeros(m, Nr)

        # Calculate the spectral grid
        self.kr = 1.0 / (rmax) * alphas

        # Calculate the spatial grid (Uniform grid with an half-cell offset)
        self.r = (rmax / Nr) * np.arange(Nr)

        # Calculate and store the inverse matrix invM
        # (imposed by the constraints on the DHT of Bessel modes)
        # NB: When compared with the FBPIC article, all the matrices here
        # are calculated in transposed form. This is done so as to use the
        # `dot` and `gemm` functions, in the `transform` method.
        self.invM = np.empty((Nr, Nr))
        denom = np.pi * rmax**2 * jn(m + 1, alphas) ** 2
        num = jn(m, self.r[:, np.newaxis] * self.kr[np.newaxis, :])
        # Get the inverse matrix
        if m != 0:
            self.invM[:, 1:] = num[:, 1:] / denom[np.newaxis, 1:]
            self.invM[:, 0] = 0.0
        else:
            self.invM[:, :] = num[:, :] / denom[np.newaxis, :]

        # Calculate the matrix M by inverting invM
        self.M = np.empty((Nr, Nr))
        if m != 0:
            self.M[1:, :] = np.linalg.pinv(self.invM[:, 1:])
            self.M[0, :] = 0.0
        else:
            self.M = np.linalg.inv(self.invM)

        self.M = to_gpu(self.M)
        self.invM = to_gpu(self.invM)
        self.r = to_gpu(self.r)
        self.kr = to_gpu(self.kr)

    def get_r(self):
        """
        Return the r grid.

        Returns
        -------
        A real 1darray containing the values of the positions
        """
        return self.r

    def get_kr(self):
        """
        Return the natural, non-uniform kr grid.

        Returns
        -------
        A real 1darray containing the values of the frequencies
        """
        return self.kr

    def transform(self, F, G):
        """
        Perform the Hankel transform of F.

        Parameters
        ----------
        F: 2darray of complex values
        Array containing the discrete values of the function for which
        the discrete Hankel transform is to be calculated.

        G: 2darray of complex values
        Array where the result will be stored
        """
        # Perform the matrix product with M
        xp.matmul(self.M, F, out=G)

    def inverse_transform(self, G, F):
        """
        Perform the inverse Hankel transform of G.

        Parameters
        ----------
        G: 2darray of real or complex values
        Array containing the values from which to compute the DHT.

        F: 2darray of real or complex values
        Array where the result will be stored.
        """
        # Perform the matrix product with invM
        xp.matmul(self.invM, G, out=F)
