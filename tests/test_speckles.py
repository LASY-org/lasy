import numpy as np
from lasy.laser import Laser
from lasy.profiles.speckled import (
    FM_SSD_Profile,
    GP_ISI_Profile,
    GP_RPM_SSD_Profile,
    PhasePlateProfile,
)
import pytest
from scipy.constants import c

def _get_arg_string(
        temporal_smoothing_type,
        speckle_args,
        ssd_args=None,
        isi_args=None,
):
    if temporal_smoothing_type.upper() in ["RPP", "CPP"]:
        args = [*speckle_args, temporal_smoothing_type]
    elif temporal_smoothing_type.upper() in ["FM SSD", "GP RPM SSD"]:
        if ssd_args is None:
            raise ValueError(f"require ssd_args for SSD smoothing")
        else:
            args = [*speckle_args, *ssd_args]
    elif temporal_smoothing_type.upper() == "GP ISI":
        if isi_args is None:
            raise ValueError(f"require isi_args for ISI smoothing")
        else:
            args = [*speckle_args, *isi_args]
    else:
        raise ValueError(f"Invalid smoothing type provided: {temporal_smoothing_type}")
    return args

def _get_laser_profile(
    temporal_smoothing_type,
    *args,
    **kw_args,
):
    if temporal_smoothing_type.upper() in ["RPP", "CPP"]:
        profile = PhasePlateProfile(*args, **kw_args)
    elif temporal_smoothing_type.upper() in "FM SSD":
        profile = FM_SSD_Profile(*args, **kw_args)
    elif temporal_smoothing_type.upper() == "GP RPM SSD":
        profile = GP_RPM_SSD_Profile(*args, **kw_args)
    elif temporal_smoothing_type.upper() == "GP ISI":
        profile = GP_ISI_Profile(*args, **kw_args)
    else:
        raise ValueError(f"Invalid smoothing type provided: {temporal_smoothing_type}")
    return profile

@pytest.mark.parametrize(
    "temporal_smoothing_type", ["RPP", "CPP", "FM SSD", "GP RPM SSD", "GP ISI"]
)
def test_intensity_distribution(temporal_smoothing_type):
    """Test whether the spatial intensity distribution and statisticis are correct.

    The distribution should be exponential, 1/<I> exp(-I/<I>) [Michel, 9.35].
    The real and imaginary parts of the envelope [Michel, Eqn. 9.26] and their product [9.30] should all be 0 on average.
    """

    wavelength = 0.351e-6  # Laser wavelength in meters
    polarization = (1, 0)  # Linearly polarized in the x direction
    laser_energy = 1.0  # J (this is the laser energy stored in the box defined by `lo` and `hi` below)
    focal_length = 3.5  # m
    beam_aperture = [0.35, 0.5]  # m
    n_beamlets = [24, 32]
    do_sinc_profile = False
    long_profile = None
    speckle_args = (
        wavelength,
        polarization,
        laser_energy,
        focal_length,
        beam_aperture,
        n_beamlets,
    )
    opt_args = {
        'do_include_transverse_envelope': do_sinc_profile,
        'long_profile': long_profile,
    }

    relative_laser_bandwidth = 0.005
    phase_modulation_amplitude = (4.1, 4.5)
    number_color_cycles = [1.4, 1.0]
    transverse_bandwidth_distribution = [1.8, 1.0]
    ssd_args = (
        relative_laser_bandwidth,
        phase_modulation_amplitude,
        number_color_cycles,
        transverse_bandwidth_distribution,
    )
    isi_args = (relative_laser_bandwidth,)

    args = _get_arg_string(temporal_smoothing_type,speckle_args,ssd_args,isi_args)
    profile = _get_laser_profile(temporal_smoothing_type, *args, **opt_args)

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


@pytest.mark.parametrize(
    "temporal_smoothing_type", ["RPP", "CPP", "FM SSD", "GP RPM SSD", "GP ISI"]
)
def test_spatial_correlation(temporal_smoothing_type):
    """Tests whether the speckles have the correct shape.

    The speckle shape is measured over one period, since the spatial profile is periodic.
    The correct speckle shape for a rectangular laser,
    determined by the autocorrelation, is the product of sinc functions [Michel, Eqn. 9.16].
    """
    wavelength = 0.351e-6  # Laser wavelength in meters
    polarization = (1, 0)  # Linearly polarized in the x direction
    laser_energy = 1.0  # J (this is the laser energy stored in the box defined by `lo` and `hi` below)
    focal_length = 3.5  # m
    beam_aperture = [0.35, 0.35]  # m
    n_beamlets = [24, 32]
    do_sinc_profile = False
    long_profile = None
    speckle_args = (
        wavelength,
        polarization,
        laser_energy,
        focal_length,
        beam_aperture,
        n_beamlets,
    )
    opt_args = {
        'do_include_transverse_envelope': do_sinc_profile,
        'long_profile': long_profile,
    }

    relative_laser_bandwidth = 0.005
    phase_modulation_amplitude = (4.1, 4.1)
    number_color_cycles = [1.4, 1.0]
    transverse_bandwidth_distribution = [1.0, 1.0]
    ssd_args = (
        relative_laser_bandwidth,
        phase_modulation_amplitude,
        number_color_cycles,
        transverse_bandwidth_distribution,
    )
    isi_args = (relative_laser_bandwidth,)

    args = _get_arg_string(temporal_smoothing_type,speckle_args,ssd_args,isi_args)
    profile = _get_laser_profile(temporal_smoothing_type, *args, **opt_args)

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


@pytest.mark.parametrize(
    "temporal_smoothing_type", ["RPP", "CPP", "FM SSD", "GP RPM SSD", "GP ISI"]
)
def test_sinc_zeros(temporal_smoothing_type):
    """Test whether the transverse sinc envelope has the correct width

    The transverse envelope for the rectangular laser has the form

    ..math::

        {\rm sinc}\left(\frac{\pi x}{\Delta x}\right)
        {\rm sinc}\left(\frac{\pi y}{\Delta y}\right)

    [Michel, Eqns. 9.11, 87, 94].
    This has widths

    ..math::

        \Delta x=\lambda_0fN_{bx}/D_x,
        \Delta y=\lambda_0fN_{by}/D_y
    """
    wavelength = 0.351e-6  # Laser wavelength in meters
    polarization = (1, 0)  # Linearly polarized in the x direction
    laser_energy = 1.0  # J (this is the laser energy stored in the box defined by `lo` and `hi` below)
    focal_length = 3.5  # m
    beam_aperture = [0.35, 0.35]  # m
    n_beamlets = [24, 48]
    do_sinc_profile = True
    long_profile = None
    speckle_args = (
        wavelength,
        polarization,
        laser_energy,
        focal_length,
        beam_aperture,
        n_beamlets,
    )
    opt_args = {
        'do_include_transverse_envelope': do_sinc_profile,
        'long_profile': long_profile,
    }

    relative_laser_bandwidth = 0.005
    phase_modulation_amplitude = (4.1, 4.1)
    number_color_cycles = [1.4, 1.0]
    transverse_bandwidth_distribution = [1.0, 1.0]
    ssd_args = (
        relative_laser_bandwidth,
        phase_modulation_amplitude,
        number_color_cycles,
        transverse_bandwidth_distribution,
    )
    isi_args = (relative_laser_bandwidth,)
    
    args = _get_arg_string(temporal_smoothing_type,speckle_args,ssd_args,isi_args)
    profile = _get_laser_profile(temporal_smoothing_type, *args, **opt_args)
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

    assert abs(F[0, :, :]).max() / abs(F).max() < 1.0e-8
    assert abs(F[-1, :, :]).max() / abs(F).max() < 1.0e-8
    assert abs(F[:, 0, :]).max() / abs(F).max() < 1.0e-8
    assert abs(F[:, -1, :]).max() / abs(F).max() < 1.0e-8


def test_FM_periodicity():
    """Test that the frequency modulated Smoothing by spectral dispersion (SSD) has the correct temporal frequency."""
    temporal_smoothing_type = 'FM SSD'
    wavelength = 0.351e-6  # Laser wavelength in meters
    polarization = (1, 0)  # Linearly polarized in the x direction
    laser_energy = 1.0  # J (this is the laser energy stored in the box defined by `lo` and `hi` below)
    focal_length = 3.5  # m
    beam_aperture = [0.35, 0.35]  # m
    n_beamlets = [24, 32]
    do_sinc_profile = False
    long_profile = None
    speckle_args = (
        wavelength,
        polarization,
        laser_energy,
        focal_length,
        beam_aperture,
        n_beamlets,
    )
    opt_args = {
        'do_include_transverse_envelope': do_sinc_profile,
        'long_profile': long_profile,
    }

    relative_laser_bandwidth = 0.005
    phase_modulation_amplitude = [4.1, 4.1]
    number_color_cycles = [1.4, 1.0]
    transverse_bandwidth_distribution = [1.0, 1.0]
    ssd_args = (
        relative_laser_bandwidth,
        phase_modulation_amplitude,
        number_color_cycles,
        transverse_bandwidth_distribution,
    )
    args = _get_arg_string(temporal_smoothing_type, speckle_args, ssd_args=ssd_args)
    laser_profile = _get_laser_profile(temporal_smoothing_type, *args, **opt_args)

    nu_laser = c / wavelength
    frac = np.sqrt(
        transverse_bandwidth_distribution[0] ** 2
        + transverse_bandwidth_distribution[1] ** 2
    )
    frac = (
        transverse_bandwidth_distribution[0] / frac,
        transverse_bandwidth_distribution[1] / frac,
    )
    phase_mod_freq = [
        relative_laser_bandwidth * sf * 0.5 / pma
        for sf, pma in zip(frac, phase_modulation_amplitude)
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
    period_error = abs(F[:, :, 0] - F[:, :, -1]).max() / abs(F).max()
    assert period_error < 1.0e-8
