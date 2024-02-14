import numpy as np

from lasy.laser import Laser
from lasy.profiles.speckle_profile import SpeckleProfile
from scipy.constants import c


def test_intensity_distribution():
    # this test seems pretty robust to any smoothing technique or physical parameters
    wavelength = 0.351e-6  # Laser wavelength in meters
    polarization = (1, 0)  # Linearly polarized in the x direction
    spot_size = 25.0e-6  # Waist of the laser pulse in meters
    pulse_duration = 30e-15  # Pulse duration of the laser in seconds
    t_peak = 0.0  # Location of the peak of the laser pulse in time
    ###
    focal_length = 3.5  # unit?
    beam_aperture = [0.35, 0.5]  # unit?
    n_beamlets = [24, 32]
    lsType = "GP ISI"
    relative_laser_bandwidth = 0.005  # unit?

    phase_mod_amp = (4.1, 4.5)
    ncc = [1.4, 1.0]
    ssd_distr = [1.8, 1.0]

    profile = SpeckleProfile(
        wavelength,
        polarization,
        spot_size,
        pulse_duration,
        t_peak,
        focal_length,
        beam_aperture,
        n_beamlets,
        lsType=lsType,
        relative_laser_bandwidth=relative_laser_bandwidth,  # 0.005
        phase_mod_amp=phase_mod_amp,
        ncc=ncc,
        ssd_distr=ssd_distr,
    )
    dimensions = "xyt"
    dx = wavelength * focal_length / beam_aperture[0]
    dy = wavelength * focal_length / beam_aperture[1]
    Lx = 1.8 * dx * n_beamlets[0]
    Ly = 3.1 * dy * n_beamlets[1]
    nu_laser = c / wavelength
    t_max = 50 / nu_laser
    lo = (0, 0, 0)
    hi = (Lx, Ly, t_max)
    num_points = (200, 250, 2)

    laser = Laser(dimensions, lo, hi, num_points, profile)

    F = laser.grid.field

    # get spatial statistics
    # <real env> = 0 = <imag env> = <er * ei>
    e_r = np.real(F)
    e_i = np.imag(F)
    er_ei = e_r * e_i
    assert np.max(abs(e_r.mean(axis=(0, 1)) / e_r.std(axis=(0, 1)))) < 1.0e-1
    assert np.max(abs(e_i.mean(axis=(0, 1)) / e_i.std(axis=(0, 1)))) < 1.0e-1
    assert np.max(abs(er_ei.mean(axis=(0, 1)) / er_ei.std(axis=(0, 1)))) < 1.0e-1

    # # compare intensity distribution with expected 1/<I> exp(-I/<I>)
    env_I = abs(F) ** 2
    I_vec = env_I.flatten()
    mean_I = I_vec.mean()
    N_hist = 200
    counts_np, bins_np = np.histogram(I_vec, bins=N_hist, density=True)
    I_dist = 1.0 / mean_I * np.exp(-bins_np / mean_I)
    error_I_dist = np.max(abs(counts_np - I_dist[:-1]))
    assert error_I_dist < 2.0e-4


def test_spatial_correlation():
    # this test seems pretty robust to any smoothing technique or physical parameters
    # provided that `Lx = dx * n_beamlets[0]` and `Ly = dy * n_beamlets[1]`
    wavelength = 0.351e-6  # Laser wavelength in meters
    polarization = (1, 0)  # Linearly polarized in the x direction
    spot_size = 25.0e-6  # Waist of the laser pulse in meters
    pulse_duration = 30e-15  # Pulse duration of the laser in seconds
    t_peak = 0.0  # Location of the peak of the laser pulse in time
    ###
    focal_length = 3.5  # unit?
    beam_aperture = [0.35, 0.35]  # unit?
    n_beamlets = [24, 32]
    lsType = "FM SSD"
    relative_laser_bandwidth = 0.005  # unit?

    phase_mod_amp = (4.1, 4.1)
    ncc = [1.4, 1.0]
    ssd_distr = [1.0, 1.0]

    profile = SpeckleProfile(
        wavelength,
        polarization,
        spot_size,
        pulse_duration,
        t_peak,
        focal_length,
        beam_aperture,
        n_beamlets,
        lsType=lsType,
        relative_laser_bandwidth=relative_laser_bandwidth,  # 0.005
        phase_mod_amp=phase_mod_amp,
        ncc=ncc,
        ssd_distr=ssd_distr,
    )
    dimensions = "xyt"
    dx = wavelength * focal_length / beam_aperture[0]
    dy = wavelength * focal_length / beam_aperture[1]
    Lx = dx * n_beamlets[0]
    Ly = dy * n_beamlets[1]
    nu_laser = c / wavelength
    tu = 1 / relative_laser_bandwidth / 50 / nu_laser
    t_max = 200 * tu
    lo = (0, 0, 0)
    hi = (Lx, Ly, t_max)
    num_points = (200, 200, 300)

    laser = Laser(dimensions, lo, hi, num_points, profile)
    F = laser.grid.field

    # compare speckle profile / autocorrelation
    # compute autocorrelation using Wiener-Khinchin Theorem

    fft_abs_all = abs(np.fft.fft2(F, axes=(0, 1))) ** 2
    ifft_abs = abs(np.fft.ifft2(fft_abs_all, axes=(0, 1))) ** 2
    acorr2_3d = np.fft.fftshift(ifft_abs, axes=(0, 1))
    acorr2_3d_norm = acorr2_3d / np.max(acorr2_3d, axis=(0, 1))

    # compare with theoretical speckle profile
    x_list = np.linspace(
        -n_beamlets[0] / 2 + 0.5, n_beamlets[0] / 2 - 0.5, num_points[0], endpoint=False
    )
    y_list = np.linspace(
        -n_beamlets[1] / 2 + 0.5, n_beamlets[1] / 2 - 0.5, num_points[1], endpoint=False
    )
    X, Y = np.meshgrid(x_list, y_list)
    acorr_theor = np.sinc(X) ** 2 * np.sinc(Y) ** 2
    error_auto_correlation = np.max(abs(acorr_theor[:, :, np.newaxis] - acorr2_3d_norm))

    assert error_auto_correlation < 5.0e-1


def test_sinc_zeros():
    # this test seems pretty robust to any smoothing technique or physical parameters
    # provided that `Lx = dx * n_beamlets[0]` and `Ly = dy * n_beamlets[1]`
    wavelength = 0.351e-6  # Laser wavelength in meters
    polarization = (1, 0)  # Linearly polarized in the x direction
    spot_size = 25.0e-6  # Waist of the laser pulse in meters
    pulse_duration = 30e-15  # Pulse duration of the laser in seconds
    t_peak = 0.0  # Location of the peak of the laser pulse in time
    ###
    focal_length = 3.5  # unit?
    beam_aperture = [0.35, 0.35]  # unit?
    n_beamlets = [24, 48]
    lsType = "GP ISI"
    relative_laser_bandwidth = 0.005  # unit?

    phase_mod_amp = (4.1, 4.5)
    ncc = [1.4, 1.0]
    ssd_distr = [1.0, 3.0]

    profile = SpeckleProfile(
        wavelength,
        polarization,
        spot_size,
        pulse_duration,
        t_peak,
        focal_length,
        beam_aperture,
        n_beamlets,
        lsType=lsType,
        relative_laser_bandwidth=relative_laser_bandwidth,  # 0.005
        phase_mod_amp=phase_mod_amp,
        ncc=ncc,
        ssd_distr=ssd_distr,
        do_include_transverse_decay=True,
    )
    dimensions = "xyt"
    dx = wavelength * focal_length / beam_aperture[0]
    dy = wavelength * focal_length / beam_aperture[1]
    Lx = dx * n_beamlets[0]
    Ly = dy * n_beamlets[1]
    nu_laser = c / wavelength
    tu = 1 / relative_laser_bandwidth / 50 / nu_laser
    t_max = 200 * tu
    lo = (-Lx, -Ly, 0)
    hi = (Lx, Ly, t_max)
    num_points = (300, 300, 10)

    laser = Laser(dimensions, lo, hi, num_points, profile)
    F = laser.grid.field

    assert np.max(abs(F[0, :, :])) < 1.0e-8
    assert np.max(abs(F[-1, :, :])) < 1.0e-8
    assert np.max(abs(F[:, 0, :])) < 1.0e-8
    assert np.max(abs(F[:, -1, :])) < 1.0e-8


def test_FM_SSD_periodicity():
    wavelength = 0.351e-6  # Laser wavelength in meters
    polarization = (1, 0)  # Linearly polarized in the x direction
    spot_size = 25.0e-6  # Waist of the laser pulse in meters
    pulse_duration = 30e-15  # Pulse duration of the laser in seconds
    t_peak = 0.0  # Location of the peak of the laser pulse in time
    ###
    focal_length = 3.5  # unit?
    beam_aperture = [0.35, 0.35]  # unit?
    n_beamlets = [24, 32]
    lsType = "FM SSD"
    relative_laser_bandwidth = 0.005  # unit?

    phase_mod_amp = (4.1, 4.1)
    ncc = [1.4, 1.0]
    ssd_distr = [1.0, 1.0]

    laser_profile = SpeckleProfile(
        wavelength,
        polarization,
        spot_size,
        pulse_duration,
        t_peak,
        focal_length,
        beam_aperture,
        n_beamlets,
        lsType=lsType,
        relative_laser_bandwidth=relative_laser_bandwidth,  # 0.005
        phase_mod_amp=phase_mod_amp,
        ncc=ncc,
        ssd_distr=ssd_distr,
    )
    nu_laser = c / wavelength
    ssd_frac = np.sqrt(ssd_distr[0] ** 2 + ssd_distr[1] ** 2)
    ssd_frac = ssd_distr[0] / ssd_frac, ssd_distr[1] / ssd_frac
    phase_mod_freq = [
        relative_laser_bandwidth * sf * 0.5 / pma
        for sf, pma in zip(ssd_frac, phase_mod_amp)
    ]
    t_max = 1.0 / phase_mod_freq[0] / nu_laser

    dimensions = "xyt"
    dx = wavelength * focal_length / beam_aperture[0]
    dy = wavelength * focal_length / beam_aperture[1]
    Lx = dx * n_beamlets[0]
    Ly = dy * n_beamlets[1]
    lo = (0, 0, 0)  # Lower bounds of the simulation box
    hi = (Lx, Ly, t_max)  # Upper bounds of the simulation box
    num_points = (160, 200, 400)  # Number of points in each dimension

    laser = Laser(dimensions, lo, hi, num_points, laser_profile)
    F = laser.grid.field
    period_error = np.max(abs(F[:, :, 0] - F[:, :, -1]))
    assert period_error < 1.0e-8


def test_temporal_correlation_ssd():
    pass
