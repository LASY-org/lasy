import numpy as np
import time
from numba import jit

from .profile import Profile


class SpeckleProfile(Profile):
    r"""
    Derived class for the analytic profile of a speckle laser pulse.

    

    More precisely, the electric field corresponds to:

    .. math::

        E_u(\\boldsymbol{x}_\\perp,t) = Re\\left[ E_0\\,
        \\exp\\left(-\\frac{\\boldsymbol{x}_\\perp^2}{w_0^2}
        - \\frac{(t-t_{peak}-ax+2ibx/w_0^2)^2}{\\tau_{eff}^2}
        - i\\omega_0(t-t_{peak}) + i\\phi_{cep}\\right) \\times p_u \\right]

    where :math:`u` is either :math:`x` or :math:`y`, :math:`p_u` is
    the polarization vector, :math:`Re` represent the real part, and
    :math:`\boldsymbol{x}_\perp` is the transverse coordinate (orthogonal
    to the propagation direction). The other parameters in this formula
    are defined below.

    Parameters
    ----------
    wavelength : float (in meter)
        The main laser wavelength :math:`\lambda_0` of the laser, which
        defines :math:`\omega_0` in the above formula, according to
        :math:`\omega_0 = 2\pi c/\lambda_0`.

    pol : list of 2 complex numbers (dimensionless)
        Polarization vector. It corresponds to :math:`p_u` in the above
        formula ; :math:`p_x` is the first element of the list and
        :math:`p_y` is the second element of the list. Using complex
        numbers enables elliptical polarizations.

    w0 : float (in meter)
        The waist of the laser pulse, i.e. :math:`w_0` in the above formula.

    tau : float (in second)
        The duration of the laser pulse, i.e. :math:`\tau` in the above
        formula. Note that :math:`\tau = \tau_{FWHM}/\sqrt{2\log(2)}`,
        where :math:`\tau_{FWHM}` is the Full-Width-Half-Maximum duration
        of the intensity distribution of the pulse.

    t_peak : float (in second)
        The time at which the laser envelope reaches its maximum amplitude,
        i.e. :math:`t_{peak}` in the above formula.

    z_foc : float (in meter), optional
        Position of the focal plane. (The laser pulse is initialized at `z=0`.)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from lasy.laser import Laser
    >>> from lasy.profiles.gaussian_profile import GaussianProfile
    >>> from lasy.utils.laser_utils import get_full_field
    >>> # Create profile.
    >>> profile = GaussianProfile(
    ...     wavelength=0.6e-6,  # m
    ...     pol=(1, 0),
    ...     laser_energy=1.,  # J
    ...     w0=5e-6,  # m
    ...     tau=30e-15,  # s
    ...     t_peak=0.  # s
    ... )
    >>> # Create laser with given profile in `rt` geometry.
    >>> laser = Laser(
    ...     dim="rt",
    ...     lo=(0e-6, -60e-15),
    ...     hi=(10e-6, +60e-15),
    ...     npoints=(50, 400),
    ...     profile=profile
    ... )
    >>> # Visualize field.
    >>> E_rt, extent = get_full_field(laser)
    >>> extent[2:] *= 1e6
    >>> extent[:2] *= 1e15
    >>> tmin, tmax, rmin, rmax = extent
    >>> vmax = np.abs(E_rt).max()
    >>> plt.imshow(
    ...     E_rt,
    ...     origin="lower",
    ...     aspect="auto",
    ...     vmax=vmax,
    ...     vmin=-vmax,
    ...     extent=[tmin, tmax, rmin, rmax],
    ...     cmap='bwr',
    ... )
    >>> plt.xlabel('t (fs)')
    >>> plt.ylabel('r (µm)')
    """

    def __init__(
        self,
        wavelength,
        pol,
        w0,
        tau,
        t_peak,
        z_foc=0,
    ):
        super().__init__(wavelength, pol)
        self.w0 = w0
        self.tau = tau
        self.t_peak = t_peak
        self.z_foc = z_foc
        self.cep_phase = 0

    def evaluate(self, x, y, t):
        """
        Return the envelope field of the laser.

        Parameters
        ----------
        x, y, t: ndarrays of floats
            Define points on which to evaluate the envelope
            These arrays need to all have the same shape.

        Returns
        -------
        envelope: ndarray of complex numbers
            Contains the value of the envelope at the specified points
            This array has the same shape as the arrays x, y, t
        """

        # focal length of the final lens, in meter
        focal_length = 7.7
        # diameter of the whole laser beam, in meter
        beam_aperture = 0.35
        # laser wave length, in meter
        wave_length = 0.351e-9
        # number of beamlets
        n_beams = [64, 64]
        # grid points in each direction
        n_grid_x, n_grid_y, n_grid_t = x.shape
        n_grid = [n_grid_x, n_grid_y]
        # n_grid = [128, 128]
        # types of smoothing. valid options are:
        # 'FM SSD', 'GS RPM SSD', 'AR RPM SSD', 'GS ISI', 'AR ISI'
        lsType = 'AR ISI'
        # if apply simple average to AR(1) to approximate Gaussian PSD
        if_sma = False
        # number of color cycles
        ncc = [1.0, 1.0]
        # (RMS value of) the amplitude of phase modulation
        beta = 4
        # bandwidth of the optics, normalized to laser frequency
        nuTotal = 0.00117
        # electric field amplitude of each beamlet, scalar or 1d numpy array
        e0 = 1.0
        # complex transform for each beamlet, scalar or 1d numpy array
        epsilon_n = 1.0
        # length of the movie, normalized to 1/omega0.
        # tMaxMovie = 2.2e5
        # time delay imposed by one echelon step in ISI, in 1/nuTotal
        tDelay = 1.5
        # delta time for the movie, in 1/omega_0, the code will round it so as
        # to complete tMax with integer steps. Increasing dt can reduce calculation
        # time. Does not apply to FM SSD in interactive plot
        # dt = 200.0
        # interactive plot or saving to files
        interactive_plot = False
        # ------------------------------------------------------------------------------
        # input ends

        # XDL unit
        xdl = wave_length / beam_aperture

        ncc = np.array(ncc)
        lsType = lsType.upper()
        # length of the time series, normalized to 1/omega0.
        # tMax = dt + tMaxMovie

        nuTotal /= np.sqrt(2)
        if beta > 0:
            nu = 0.5 * nuTotal / beta
        else:
            nu = 0
        # s is the parameter for gratings in SSD. equal to time delay in xdl units
        if nu > 0:
            s = np.divide(2 * np.pi * ncc, nu)
        else:
            s = [0.0, 0.0]

        # RPP
        # phi_n = np.pi * np.random.binomial(1, 0.5, (n_beams[0], n_beams[1]))
        # CPP
        phi_n = np.pi * np.random.uniform(-np.pi, np.pi, (n_beams[0], n_beams[1]))
        # x0, x1 are normalized to the beam aperture
        x0, x1 = np.meshgrid(np.linspace(-0.5, 0.5, num=n_beams[0]),
                            np.linspace(-0.5, 0.5, num=n_beams[1]))
        
        def general_form_beamlets_2d(amp, trans_n, psi_n, ps_n):
            """ General form of the beamlets (1d version).

            E0(x,t)=amp \sum\limits_n e^{i \psi_n} \trans_n \exp[i \ps_n]
            :param amp: field amplitude of the beamlets
            :param trans_n: quantities that define complex transformation for beamlets
            :param psi_n: describe the phase and temporal bandwidth of each beamlet
            :param ps_n: phase shift of each beamlet due to phase plate
            :return: full beam consist of all beamlets
            """
            beamlets = amp * trans_n * np.exp(1j * (psi_n + ps_n))
            return beamlets

        def ssd_2d_fm(t):
            """ Beamlets after SSD and before the final focal len (2d version).

            :param t: current time
            :return: near field electric field amplitude of the full beam
            """
            psi_n = beta * (np.sin(nu * (t + s[0] * x0)) +
                            np.sin(nu * (t + s[1] * x1)))
            return general_form_beamlets_2d(e0, epsilon_n, psi_n, phi_n)

        # time delay array for beamlets
        # needed for phase modulation
        # tn_d = np.arange(0.0, n_beams[0] * n_beams[1]).reshape(n_beams)
        # tn = np.long(0)

        dx = np.divide(2 * np.pi, n_grid)
        xlp0, xlp1 = np.meshgrid(np.linspace(-0.5 * n_beams[0], 0.5 * n_beams[0],
                                            num=n_beams[0]),
                                np.linspace(-0.5 * n_beams[1], 0.5 * n_beams[1],
                                            num=n_beams[1]))


        laser_smoothing_2d = ssd_2d_fm
        # pmPhase = None

        gn = n_grid
        xfp0, xfp1 = np.meshgrid(np.linspace(-0.5 * gn[0], 0.5 * gn[0], num=gn[0]),
                                np.linspace(-0.5 * gn[1], 0.5 * gn[1], num=gn[1]))
        # constant phase shift due to beam propagation
        proPhase = np.exp(1j * (np.square(xfp0) + np.square(xfp1))
                        * xdl * focal_length / beam_aperture * np.pi +
                        2j * np.pi * focal_length / wave_length)
        
       
        @jit()
        def focal_len_expensive_part(beamlets, field,xbound,ybound):
            for ibx in range(xbound[0],xbound[1]):
                for iby in range(ybound[0],ybound[1]):
                    field[ibx, iby] = np.sum(np.multiply(
                        np.exp(1j * (ibx * dx[0] * xlp0 + iby * dx[1] * xlp1)),
                        beamlets))
                    # field[ibx,iby] = 1j
            return field
        
        @jit()
        def focal_len_expensive_precompute(beamlets, field, xbound, ybound):

            xmat = np.exp(1j*dx[0] * xlp0)
            ymat = np.exp(1j*dx[1] * xlp1)
            for ibx in range(xbound[0],xbound[1]):
                for iby in range(ybound[0],ybound[1]):
                    field[ibx, iby] = np.sum(np.multiply(
                        np.power(xmat,ibx) * np.power(ymat,iby),
                        beamlets))
            return field
        
        def focal_len_2d(beamlets):
            """ Use the diffraction integral to calculate the interference of beamlets on focal plane (2d version).

            :param beamlets: electric field of full beam
            :return: far fields pattern on the focal plane
            """
            field = np.zeros(n_grid, dtype=complex)
            if n_beams[0] == n_grid[0] and n_beams[1] == n_grid[1]:
                field = np.fft.fft2(beamlets)
            else:
                # naive sum to calculate the Fourier transform
                # field[:,:] = np.sum( np.multiply)

                # original
                # for ibx in range(int(-n_grid[0] / 2), int(n_grid[0] - n_grid[0] / 2)):
                #     for iby in range(int(-n_grid[1] / 2), int(n_grid[1] - n_grid[1] / 2)):
                #         field[ibx, iby] = np.sum(np.multiply(
                #             np.exp(1j * (ibx * dx[0] * xlp0 + iby * dx[1] * xlp1)),
                #             beamlets))

                # trying to precompute
                # xmat = np.exp(1j*dx[0] * xlp0)
                # ymat = np.exp(1j*dx[1] * xlp1)
                # for ibx in range(int(-n_grid[0] / 2), int(n_grid[0] - n_grid[0] / 2)):
                #     for iby in range(int(-n_grid[1] / 2), int(n_grid[1] - n_grid[1] / 2)):
                #         field[ibx, iby] = np.sum(np.multiply(
                #             np.power(xmat,ibx) * np.power(ymat,iby),
                #             beamlets))
                        
                # trying numba
                xbound = (int(-n_grid[0] / 2), int(n_grid[0] - n_grid[0] / 2))
                ybound= (int(-n_grid[1] / 2), int(n_grid[1] - n_grid[1] / 2))
                field = focal_len_expensive_part(beamlets, field, xbound, ybound)
                # field = focal_len_expensive_precompute(beamlets, field, xbound, ybound)
            field = np.multiply(proPhase, np.fft.fftshift(field))
            return field

        envelope = np.zeros(x.shape, dtype=complex)
        for i, t_i in enumerate(t[0,0]):
            if i % 100 == 0:
                print(f'ti={t_i:.3e} s')
            t2 = time.time()
            beamlets = laser_smoothing_2d(t_i)
            t3 = time.time()
            fp_speckle = focal_len_2d(beamlets)
            t4 = time.time()
            print(f'beamlet init took {t3-t2:.3e} s')
            print(f'focal len took {t4-t3:.3e} s')
            envelope[:,:,i] = fp_speckle


        ###############

        # it is possible we ignore this
        spacetime = np.exp(-( (t - self.t_peak)** 2) / self.tau **2)

        # not sure about this
        oscillatory = np.exp(1.0j * (self.cep_phase - self.omega0 * (t - self.t_peak))) 

        #envelope *= spacetime * oscillatory

        return envelope