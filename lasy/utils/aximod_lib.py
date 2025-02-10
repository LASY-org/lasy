import numpy as np
from scipy.constants import c

from axiprop.steppers import StepperNonParaxial
from axiprop.common import CommonTools

try:
    from tqdm.auto import tqdm
    tqdm_available = True
    bar_format='{l_bar}{bar}| {elapsed}<{remaining} [{rate_fmt}{postfix}]'
except Exception:
    tqdm_available = False

class StepperNonParaxial_mod(StepperNonParaxial):
    def __init__(self):
        super().__init__()

    def t2z_slice(self, u, z_axis=None, slice_ind=None, slice_axis=0, z0=0.0, t0=0.0, show_progress=True):
        """
        Reconstruct wave `u` in the spatial domain.

        Parameters
        ----------
        u: 2darray of complex or double
            Spectral-radial distribution of the field to propagate.

        z_axis: array of floats (m)
            Axis over which wave should be reconstructed.

        slice_ind: int or None
            index of values to take from slice_axis. If None then the middle will be taken.

        slice_axis: int 
            axis to take slice of (this is the dimension that is lost)
            

        Returns
        -------
        u: 3darray of complex or double
            Array with the steps of the reconstructed field.
        """
        assert u.dtype == self.dtype

        Nsteps = len(z_axis)
        if Nsteps==0:
            return None

        if slice_ind is None:
            slice_ind = int(u.shape[slice_axis]/2) # take the middle of the slice axis by default
        
        u_steps = np.zeros( (Nsteps, self.shape_trns_new[0],self.Nkz),
                            dtype=u.dtype)

        if tqdm_available and show_progress:
            pbar = tqdm(total=self.Nkz*Nsteps, bar_format=bar_format)

        for ikz in range(self.Nkz):
            self.u_loc = self.bcknd.to_device(u[ikz].copy())
            self.TST()
            k_loc = self.bcknd.sqrt(self.bcknd.abs( self.kz[ikz]**2 - \
                                                     self.kr2 ))

            u_ht0 = self.u_ht.copy() * np.exp( 1j * self.kz[ikz] * c * t0 )
            for i_step in range(Nsteps):
                self.u_ht[:] = u_ht0 * \
                    self.bcknd.exp( 1j * (z_axis[i_step]-z0) * k_loc )
                self.iTST()
                u_steps[i_step,:, ikz] = np.take(self.bcknd.to_host(self.u_iht),slice_ind,slice_axis)

                if tqdm_available and show_progress:
                    pbar.update(1)
                elif show_progress and not tqdm_available:
                    print(f"Done step {i_step} of {Nsteps} "+ \
                          f"for wavelength {ikz+1} of {self.Nkz}",
                          end='\r', flush=True)

        if tqdm_available and show_progress:
            pbar.close()

        return u_steps

class PropagatorFFT2_mod(CommonTools, StepperNonParaxial_mod):
    """
    Class for the propagator with two-dimensional Fast Fourier transform (FFT2)
    for TST.

    Contains methods to:
    - setup TST data buffers;
    - perform a forward FFT;
    - perform a inverse FFT;
    """

    def __init__(self, x_axis, y_axis, kz_axis,
                 dtype=np.complex128, backend=None,
                 verbose=True):
        """
        Construct the propagator.

        Parameters
        ----------
        x_axis: tuple (Lx, Nx)
          Define the x-axis grid with parameters:
            Lx: float (m)
                Full size of the calculation domain along x-axis.

            Nx: int
                Number of nodes of the x-grid. Better be an odd number,
                in order to make a symmteric grid.

        y_axis: tuple (Ly, Ny)
          Define the y-axis grid with parameters:
            Ly: float (m)
                Full size of the calculation domain along y-axis.

            Ny: int
                Number of nodes of the y-grid.Better be an odd number,
                in order to make a symmteric grid.

        kz_axis: a tuple (k0, Lkz, Nkz) or a 1D numpy.array
            When tuple is given the axis is created using:

              k0: float (1/m)
                Central wavenumber of the spectral domain.

              Lkz: float (1/m)
                Total spectral width in units of wavenumbers.

              Nkz: int
                Number of spectral modes (wavenumbers) to resolve the temporal
                profile of the wave.

        dtype: type (optional)
            Data type to be used. Default is np.complex128.

        backend: string
            Backend to be used. See axiprop.backends.AVAILABLE_BACKENDS for the
            list of available options.
        """
        self.dtype = dtype

        self.init_backend(backend, verbose)
        self.init_kz(kz_axis)
        self.x, self.y, self.r, self.r2 = self.init_xy_uniform(x_axis, y_axis)
        self.init_kxy_uniform(self.x, self.y)
        self.init_TST()

    def init_TST(self):
        """
        Setup data buffers for TST.
        """
        Nx = self.x.size
        Ny = self.y.size
        dtype = self.dtype

        self.shape_trns = (Nx, Ny)
        self.shape_trns_new = (Nx, Ny)

        self.u_loc = self.bcknd.zeros((Nx, Ny), dtype)
        self.u_ht = self.bcknd.zeros((Nx, Ny), dtype)
        self.u_iht = self.bcknd.zeros((Nx, Ny), dtype)

        self.fft2, self.ifft2, fftshift = self.bcknd.make_fft2(self.u_iht, self.u_ht)

    def TST(self):
        """
        Forward FFT transform.
        """
        self.u_ht = self.fft2(self.u_loc, self.u_ht)

    def iTST(self):
        """
        Inverse FFT transform.
        """
        self.u_iht = self.ifft2(self.u_ht, self.u_iht)




