import matplotlib.pyplot as plt
import numpy as np
import skimage
from mpl_toolkits.axes_grid1 import make_axes_locatable

from lasy.profiles.combined_profile import CombinedLongitudinalTransverseProfile
from lasy.profiles.longitudinal.gaussian_profile import GaussianLongitudinalProfile
from lasy.profiles.transverse.hermite_gaussian_profile import (
    HermiteGaussianTransverseProfile,
)
from lasy.profiles.transverse.transverse_profile_from_data import (
    TransverseProfileFromData,
)
from lasy.utils.mode_decomposition import hermite_gauss_decomposition

# Define the transverse profile of the laser pulse
img_url = "https://user-images.githubusercontent.com/27694869/228038930-d6ab03b1-a726-4b41-a378-5f4a83dc3778.png"
intensityData = skimage.io.imread(img_url)
intensityData[intensityData < 2.1] = 0
pixel_calib = 0.186e-6
lo = (
    -intensityData.shape[0] / 2 * pixel_calib,
    -intensityData.shape[1] / 2 * pixel_calib,
)
hi = (
    intensityData.shape[0] / 2 * pixel_calib,
    intensityData.shape[1] / 2 * pixel_calib,
)
energy = 0.5
pol = (1, 0)
transverse_profile = TransverseProfileFromData(intensityData, lo, hi)

# Define longitudinal profile of the laser pulse
wavelength = 800e-9
tau = 30e-15
t_peak = 0.0
longitudinal_profile = GaussianLongitudinalProfile(wavelength, tau, t_peak)

# Combine into full laser profile
profile = CombinedLongitudinalTransverseProfile(
    wavelength, pol, energy, longitudinal_profile, transverse_profile
)

# Calculate the decomposition into hermite-gauss modes
n_x_max = 20
n_y_max = 20
modeCoeffs, waist = hermite_gauss_decomposition(
    transverse_profile, n_x_max=n_x_max, n_y_max=n_y_max, res=0.5e-6
)

# Reconstruct the pulse using a series of hermite-gauss modes
for i, mode_key in enumerate(list(modeCoeffs)):
    tmp_transverse_profile = HermiteGaussianTransverseProfile(
        waist, mode_key[0], mode_key[1]
    )
    if i == 0:
        reconstructedProfile = modeCoeffs[
            mode_key
        ] * CombinedLongitudinalTransverseProfile(
            wavelength, pol, energy, longitudinal_profile, tmp_transverse_profile
        )
    else:
        reconstructedProfile += modeCoeffs[
            mode_key
        ] * CombinedLongitudinalTransverseProfile(
            wavelength, pol, energy, longitudinal_profile, tmp_transverse_profile
        )

# Plotting the results
x = np.linspace(-5 * waist, 5 * waist, 500)
X, Y = np.meshgrid(x, x)

fig, ax = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)

pltextent = (np.min(x) * 1e6, np.max(x) * 1e6, np.min(x) * 1e6, np.max(x) * 1e6)
prof1 = np.abs(profile.evaluate(X, Y, 0)) ** 2
divider0 = make_axes_locatable(ax[0])
ax0_cb = divider0.append_axes("right", size="5%", pad=0.05)
pl0 = ax[0].imshow(prof1, cmap="magma", extent=pltextent, vmin=0, vmax=np.max(prof1))
cbar0 = fig.colorbar(pl0, cax=ax0_cb)
cbar0.set_label("Intensity (norm.)")
ax[0].set_xlabel("x (micron)")
ax[0].set_ylabel("y (micron)")
ax[0].set_title("Original Profile")

prof2 = np.abs(reconstructedProfile.evaluate(X, Y, 0)) ** 2
divider1 = make_axes_locatable(ax[1])
ax1_cb = divider1.append_axes("right", size="5%", pad=0.05)
pl1 = ax[1].imshow(prof2, cmap="magma", extent=pltextent, vmin=0, vmax=np.max(prof1))
cbar1 = fig.colorbar(pl1, cax=ax1_cb)
cbar1.set_label("Intensity (norm.)")
ax[1].set_xlabel("x (micron)")
ax[1].set_ylabel("y (micron)")
ax[1].set_title("Reconstructed Profile")


prof3 = (prof1 - prof2) / np.max(prof1)
divider2 = make_axes_locatable(ax[2])
ax2_cb = divider2.append_axes("right", size="5%", pad=0.05)
pl2 = ax[2].imshow(100 * np.abs(prof3), cmap="magma", extent=pltextent, vmin=0, vmax=2)
cbar2 = fig.colorbar(pl2, cax=ax2_cb)
cbar2.set_label("|Error| (%)")
ax[2].set_xlabel("x (micron)")
ax[2].set_ylabel("y (micron)")
ax[2].set_title("Error")

fig.suptitle(
    "Hermite-Gauss Reconstruction using n_x_max = %i, n_y_max = %i" % (n_x_max, n_y_max)
)
plt.show()
