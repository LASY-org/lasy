import numpy as np

from lasy.utils.zernike import zernike

from .optical_element import OpticalElement


class ZernikeAberrations(OpticalElement):
    r"""
    Class for an optic applying a set of Zernike aberrations.

    More precisely, the amplitude multiplier corresponds to:

    .. math::

        T(\boldsymbol{x}_\perp,\omega) = \exp( i \sum_j a_j Z_j(\boldsymbol{x}_\perp))

    where
    :math:`\boldsymbol{x}_\perp` is the transverse coordinate (orthogonal
    to the propagation direction). :math:`Z_j` is the j-th Zernike polynomial,
    ordered according the OSA/ANSI indexing. The Zernike polynomials are normalized
    such that their integral over the unit disk is equal to :math:`\pi`. In the above
    formula, the total phase added to the pulse is a weighted sum of these Zernike
    polynomials with weights :math:`a_j`.

    For more information see: https://en.wikipedia.org/wiki/Zernike_polynomials


    Parameters
    ----------
    pupil_coords : tuple of floats (meters)
        A tuple of floats (cgx,cgy,r) with the first two elements corresponding to the center
        point and third element the radius of the pupil on which the zernike polynomial is
        defined.
    zernike_amplitudes : dict
        A dictionary with integer keys representing the OSA/ANSI indexing of the
        individual Zernike Polynomials. The values corresponding to these keys
        are floats giving the amplitudes / weights of the relevant Zernike polynomials.

    """

    def __init__(self, pupil_coords, zernike_amplitudes):
        self.pupil_coords = pupil_coords
        self.zernike_amplitudes = zernike_amplitudes

    def amplitude_multiplier(self, x, y, omega):
        """
        Return the amplitude multiplier.

        Parameters
        ----------
        x, y, omega : ndarrays of floats
            Define points on which to evaluate the multiplier.
            These arrays need to all have the same shape.

        Returns
        -------
        multiplier : ndarray of complex numbers
            Contains the value of the multiplier at the specified points.
            This array has the same shape as the array omega.
        """
        rr = np.sqrt(x**2 + y**2)
        phase = np.zeros_like(rr)

        for j in list(self.zernike_amplitudes):
            # Create the zernike phase and ensure it has the same number of dimensions as phase
            zernike_phase = zernike(x[..., 0], y[..., 0], self.pupil_coords, j)[
                ..., None
            ]  # Expand last axis

            # Increase the length of the frequency dimension such that the shape is suitable to be added
            # to the phase array, then add it
            phase += self.zernike_amplitudes[j] * np.broadcast_to(
                zernike_phase, phase.shape
            )

        return np.exp(1j * phase)
