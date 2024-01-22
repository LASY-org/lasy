import numpy as np

from lasy.laser import Laser
from lasy.profiles.speckle_profile import SpeckleProfile
from lasy.utils.laser_utils import get_spectrum, compute_laser_energy, get_duration


# tolerances are currently hard-coded and generous.
# I don't know what best practice is here

def test1():
    # these tests could all be done at many points in time, not just one
    # should we loop?

    # initialize SSD profile
    wavelength     = 800e-9  # Laser wavelength in meters
    polarization   = (1,0)   # Linearly polarized in the x direction
    spot_size      = 25e-6   # Waist of the laser pulse in meters
    pulse_duration = 30e-15  # Pulse duration of the laser in seconds
    t_peak         = 0.0     # Location of the peak of the laser pulse in time
    focal_length   = 7.7
    beam_aperture  = 0.35
    n_beams        = [64, 64]
    beta = 4
    # bandwidth of the optics, normalized to laser frequency
    nuTotal = 0.00117
    nuTotal /= np.sqrt(2)

    nu = 0.5 * nuTotal / beta
    f = nu/(2*np.pi)
    t_max = 1/f  # one period of SSD

    profile = SpeckleProfile(
        wavelength,
        polarization,
        spot_size,
        pulse_duration,
        t_peak,
        focal_length,
        beam_aperture,
        n_beams)
    
    dimensions     = 'xyt'                              
    lo             = (-5*spot_size,-5*spot_size,0)
    hi             = (5*spot_size,5*spot_size,t_max)
    num_points     = (300,300,20)

    laser = Laser(dimensions,lo,hi,num_points,profile)

    x = np.linspace(lo[0], hi[0], num_points[0])
    y = np.linspace(lo[1], hi[1], num_points[1])
    t = np.linspace(lo[2], hi[2], num_points[2])
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    F = profile.evaluate(X, Y, T)

    # test periodicity of SSD
    assert np.max(abs(F[:,:,-1] - F[:,:,0])) < 1e-12

    # get spatial statistics
    tind = 0
    env = F[:,:,tind]
    # <real env> = 0 = <imag env> = <er * ei>
    e_r = np.real(env)
    e_i = np.imag(env)
    er_ei = e_r * e_i
    assert e_r.mean() / e_r.std() < 1.e-2
    assert e_i.mean() / e_i.std() < 1.e-2
    assert er_ei.mean() / er_ei.std() < 1.e-1


    # compare intensity distribution with expected 1/<I> exp(-I/<I>)
    env_I = abs(env)**2
    I_vec = env_I.flatten()
    mean_I = I_vec.mean()
    N_hist = 100
    counts_np, bins_np = np.histogram(I_vec,bins=N_hist,density=True)
    I_dist = 1./mean_I * np.exp(-bins_np/mean_I)
    error_I_dist = np.max(abs(counts_np - I_dist[1:]))
    assert error_I_dist < 1.e-4

    # compare speckle profile / autocorrelation
    # compute autocorrelation using Wiener-Khinchin Theorem
    fft_abs = abs(np.fft.fft2(env))**2
    ifft_abs = abs(np.fft.ifft2(fft_abs))**2
    auto_correlation = np.fft.fftshift(ifft_abs)
    acorr_norm = auto_correlation / np.max(auto_correlation)
    # compare with theoretical speckle profile
    x_list = np.linspace(-n_beams[0] / 2 + 0.5, n_beams[0] / 2 - 0.5,num_points[0],endpoint=False)
    y_list = np.linspace(-n_beams[1] / 2 + 0.5, n_beams[1] / 2 - 0.5,num_points[1],endpoint=False)
    X,Y = np.meshgrid(x_list,y_list)
    acorr_theor = np.sinc(X)**2 * np.sinc(Y)**2
    error_auto_correlation = np.max(abs(acorr_norm - acorr_theor))
    assert error_auto_correlation < 1.e-1


