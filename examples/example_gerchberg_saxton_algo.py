from lasy.profiles.gaussian_profile import GaussianProfile
from lasy.laser import Laser
from lasy.utils.phase_retrieval import gerchberg_saxton_algo
import matplotlib.pyplot as plt
import numpy as np
from lasy.utils.zernike import zernike
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable


# DEFINE PHYSICAL PARAMETERS & CREATE LASER PROFILE
wavelength = 800e-9
pol = (1, 0)
laser_energy = 1e-3
w0 = 25e-6
tau = 30e-15
t_peak = 0
pulseProfile = GaussianProfile(wavelength, pol, laser_energy, w0, tau, t_peak)

# DEFNIE COMPUTATION PARAMETERS AND CREATE LASER OBJECT
dim = "xyt"
lo = (-75e-6, -75e-6, -50e-15)
hi = (75e-6, 75e-6, 50e-15)
npoints = (100, 100, 100)
laser = Laser(dim=dim, lo=lo, hi=hi, npoints=npoints, profile=pulseProfile)

# CALCULATE THE REQUIRED PHASE ABERRATION
x = np.linspace(lo[0], hi[0], npoints[0])
y = np.linspace(lo[1], hi[1], npoints[1])
X, Y = np.meshgrid(x, y)
pupilRadius = 2 * w0
phase = -0.2 * zernike(X, Y, (0, 0, pupilRadius), 3)

R = np.sqrt(X**2 + Y**2)
phaseMask = np.ones_like(phase)
phaseMask[R > pupilRadius] = 0

# NOW ADD THE PHASE TO EACH SLICE OF THE FOCUS
phase3D = np.repeat(phase[:, :, np.newaxis], npoints[2], axis=2)
laser.field.field = np.abs(laser.field.field) * np.exp(1j * phase3D)

# PROPAGATE THE FIELD FIELD FOWARDS AND BACKWARDS BY 1 MM
propDist = 2e-3
laserForward = copy.deepcopy(laser)
laserForward.propagate(propDist)
laserBackward = copy.deepcopy(laser)
laserBackward.propagate(-propDist)

# PERFORM GERCHBERG-SAXTON ALGORTIHM TO RETRIEVE PHASE
phaseBackward, phaseForward, amp_error = gerchberg_saxton_algo(
    laserBackward,
    laserForward,
    2 * propDist,
    condition="amplitude_error",
    max_itterations=50,
    amplitude_error=1e-6,
    debug=True,
)

# GET THE FIELD AND PLOT IT
fig, ax = plt.subplots(2, 5, figsize=(15, 5), tight_layout=True)
tIndx = 50
extent = (lo[0] * 1e6, hi[0] * 1e6, lo[1] * 1e6, hi[1] * 1e6)


def addColorbar(im, ax, label=None):
    """Create a colorbar and add it to the plot."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    cax.set_ylabel(label)


im0 = ax[0, 0].imshow(
    np.abs(laser.field.field[:, :, tIndx]) ** 2, extent=extent, cmap="PuRd"
)
addColorbar(im0, ax[0, 0], "Intensity (norm.)")
ax[0, 0].set_title("Inten. z = 0.0 mm")
ax[0, 0].set_xlabel("x ($\mu m$)")
ax[0, 0].set_ylabel("y ($\mu m$)")


im1 = ax[0, 1].imshow(
    np.angle(laser.field.field[:, :, tIndx]) * phaseMask, extent=extent, cmap="coolwarm"
)
addColorbar(im1, ax[0, 1], "Phase (rad.)")
ax[0, 1].set_title("Phase z = 0.0 mm")
ax[0, 1].set_xlabel("x ($\mu m$)")
ax[0, 1].set_ylabel("y ($\mu m$)")


laser_calc = copy.deepcopy(laserBackward)
laser_calc.field.field = np.abs(laser_calc.field.field) * np.exp(1j * phaseBackward)
laser_calc.propagate(propDist)
im2 = ax[1, 0].imshow(
    np.abs(
        np.abs(laser.field.field[:, :, tIndx]) ** 2
        - np.abs(laser_calc.field.field[:, :, tIndx]) ** 2
    ),
    extent=extent,
    cmap="PuRd",
)
ax[1, 0].set_title("Inten. Res. z = 0.0 mm")
ax[1, 0].set_xlabel("x ($\mu m$)")
ax[1, 0].set_ylabel("y ($\mu m$)")
addColorbar(im2, ax[1, 0], "Intensity (norm.)")

phaseResidual = np.angle(laser_calc.field.field[:, :, tIndx]) - np.angle(
    laser.field.field[:, :, tIndx]
)
phaseResidual -= phaseResidual[int(npoints[1] / 2), int(npoints[0] / 2)]
maxPhaseRes = np.max(np.abs(phaseResidual) * phaseMask)
im3 = ax[1, 1].imshow(
    phaseResidual * phaseMask,
    extent=extent,
    cmap="coolwarm",
    vmin=-maxPhaseRes,
    vmax=maxPhaseRes,
)
ax[1, 1].set_title("Phase Res. z = 0.0 mm")
ax[1, 1].set_xlabel("x ($\mu m$)")
ax[1, 1].set_ylabel("y ($\mu m$)")
addColorbar(im3, ax[1, 1], "Phase (rad.)")


im4 = ax[0, 2].imshow(
    np.abs(laserBackward.field.field[:, :, tIndx]) ** 2, extent=extent, cmap="PuRd"
)
addColorbar(im4, ax[0, 2], "Intensity (norm.)")
ax[0, 2].set_title("Inten. z = %.1f mm" % (-propDist * 1e3))
ax[0, 2].set_xlabel("x ($\mu m$)")
ax[0, 2].set_ylabel("y ($\mu m$)")

im5 = ax[0, 3].imshow(
    np.angle(laserBackward.field.field[:, :, tIndx]), extent=extent, cmap="coolwarm"
)
addColorbar(im5, ax[0, 3], "Phase (rad.)")
ax[0, 3].set_title("Phase z = %.1f mm" % (-propDist * 1e3))
ax[0, 3].set_xlabel("x ($\mu m$)")
ax[0, 3].set_ylabel("y ($\mu m$)")

phaseResidual = (
    np.angle(laserBackward.field.field[:, :, tIndx]) - phaseBackward[:, :, tIndx]
)
phaseResidual -= phaseResidual[int(npoints[1] / 2), int(npoints[0] / 2)]
maxPhaseRes = np.max(np.abs(phaseResidual) * phaseMask)
im6 = ax[0, 4].imshow(
    phaseResidual * phaseMask,
    extent=extent,
    cmap="coolwarm",
    vmin=-maxPhaseRes,
    vmax=maxPhaseRes,
)
addColorbar(im6, ax[0, 4], "Phase (rad.)")
ax[0, 4].set_title("Phase Res. z = %.1f mm" % (-propDist * 1e3))
ax[0, 4].set_xlabel("x ($\mu m$)")
ax[0, 4].set_ylabel("y ($\mu m$)")

im7 = ax[1, 2].imshow(
    np.abs(laserForward.field.field[:, :, tIndx]) ** 2, extent=extent, cmap="PuRd"
)
addColorbar(im7, ax[1, 2], "Intensity (norm.)")
ax[1, 2].set_title("Inten. z = %.1f mm" % (propDist * 1e3))
ax[1, 2].set_xlabel("x ($\mu m$)")
ax[1, 2].set_ylabel("y ($\mu m$)")
im8 = ax[1, 3].imshow(
    np.angle(laserForward.field.field[:, :, tIndx]), extent=extent, cmap="coolwarm"
)
addColorbar(im8, ax[1, 3], "Phase (rad.)")
ax[1, 3].set_title("Phase z = %.1f mm" % (propDist * 1e3))
ax[1, 3].set_xlabel("x ($\mu m$)")
ax[1, 3].set_ylabel("y ($\mu m$)")

phaseResidual = (
    np.angle(laserForward.field.field[:, :, tIndx]) - phaseForward[:, :, tIndx]
)
phaseResidual -= phaseResidual[int(npoints[1] / 2), int(npoints[0] / 2)]
maxPhaseRes = np.max(np.abs(phaseResidual) * phaseMask)
im9 = ax[1, 4].imshow(
    phaseResidual * phaseMask,
    extent=extent,
    cmap="coolwarm",
    vmin=-maxPhaseRes,
    vmax=maxPhaseRes,
)
addColorbar(im9, ax[1, 4], "Phase (rad.)")
ax[1, 4].set_title("Phase Res. z = %.1f mm" % (propDist * 1e3))
ax[1, 4].set_xlabel("x ($\mu m$)")
ax[1, 4].set_ylabel("y ($\mu m$)")
plt.show()
