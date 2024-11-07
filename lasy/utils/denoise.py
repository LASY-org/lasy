from lasy.profiles.combined_profile import CombinedLongitudinalTransverseProfile
from lasy.profiles.transverse.hermite_gaussian_profile import (
    HermiteGaussianTransverseProfile,
)
from lasy.utils.mode_decomposition import hermite_gauss_decomposition


<<<<<<< Updated upstream
def denoise_laser(parameters, n_modes_x=2, n_modes_y=2):
    transverse_profile = parameters["transverse_profile"]
    longitudinal_profile = parameters["longitudinal_profile"]
    polarization = parameters["polarization"]
    laser_energy_new = 0

    if parameters.get("wavelength") is None:
        wavelength = longitudinal_profile.lambda0
    else:
        wavelength = parameters["wavelength"]

    if parameters.get("laser_energy") is None:
        laser_energy = 1  # In joules
    else:
        laser_energy = parameters["laser_energy"]

    if parameters.get("resolution") is None:
        resolution = 0.2e-6
    else:
        resolution = parameters["resolution"]
=======
class DenoisedLaser:
    def denoise_lg_modes(parameters, n_modes_x = 2, n_modes_y = 2):
        transverse_profile = parameters["transverse_profile"]
        longitudinal_profile = parameters["longitudinal_profile"]
        polarization = parameters["polarization"]
        laser_energy_new = 0
        
        if parameters.get("wavelength") is None:
            try:
                wavelength = transverse_profile.lambda0
            except AttributeError:
                raise ValueError("The wavelength must be specified.")
        else:
            wavelength = parameters["wavelength"]
        
        if parameters.get("laser_energy") is None:
            laser_energy = 1 # In joules
        else:
            laser_energy = parameters["laser_energy"]
        
        if parameters.get("resolution") is None:
            resolution = 0.2e-6
        else:
            resolution = parameters["resolution"]
>>>>>>> Stashed changes

        # Calculate the decomposition and waist of the laser pulse
        modeCoeffs, waist = hermite_gauss_decomposition(
            transverse_profile, n_modes_x, n_modes_y, resolution
        )

<<<<<<< Updated upstream
    # Denosing the laser profile
    for i, mode_key in enumerate(list(modeCoeffs)):
        tmp_transverse_profile = HermiteGaussianTransverseProfile(
            waist, mode_key[0], mode_key[1]
        )
        print(f"Mode {i}: {mode_key} with coefficient {modeCoeffs[mode_key]}")
        laser_energy_new += modeCoeffs[mode_key] ** 2  # Energy fraction of the mode
        if i == 0:  # First mode (0,0)
            laser_profile_cleaned = modeCoeffs[
                mode_key
            ] * CombinedLongitudinalTransverseProfile(
=======
        # Denosing the laser profile
        for i, mode_key in enumerate(list(modeCoeffs)):
            tmp_transverse_profile = HermiteGaussianTransverseProfile(
                waist, mode_key[0], mode_key[1]
            )
            print(f"Mode {i}: {mode_key} with coefficient {modeCoeffs[mode_key]}")
            laser_energy_new += modeCoeffs[mode_key] ** 2  # Energy fraction of the mode
            if i == 0:  # First mode (0,0)
                laser_profile_cleaned = modeCoeffs[
                    mode_key
                ] * CombinedLongitudinalTransverseProfile(
>>>>>>> Stashed changes
                wavelength,
                polarization,
                laser_energy,
                longitudinal_profile,
                tmp_transverse_profile,
            )
<<<<<<< Updated upstream
        else:  # All other modes
            laser_profile_cleaned += modeCoeffs[
                mode_key
            ] * CombinedLongitudinalTransverseProfile(
=======
            else:  # All other modes
                laser_profile_cleaned += modeCoeffs[
                    mode_key
                ] * CombinedLongitudinalTransverseProfile(
>>>>>>> Stashed changes
                wavelength,
                polarization,
                laser_energy,
                longitudinal_profile,
                tmp_transverse_profile,
            )
<<<<<<< Updated upstream
    # Energy loss due to decomposition
    energy_loss = 1 - laser_energy_new
    print(f"Energy loss: {energy_loss * 100:.2f}%")
    return laser_profile_cleaned, laser_energy_new
=======
        # Energy loss due to decomposition
        energy_loss = 1 - laser_energy_new
        print(f"Energy loss: {energy_loss * 100:.2f}%")
        return laser_profile_cleaned, laser_energy_new
    
    def help():
        print("Please refer to the following tutorial: https://lasydoc.readthedocs.io/en/latest/tutorials/denoised_laser.html")
>>>>>>> Stashed changes
