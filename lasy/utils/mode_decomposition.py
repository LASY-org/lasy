from lasy.profiles.transverse.transverse_profile import TransverseProfile
from lasy.profiles.transverse.transverse_profile_from_data import TransverseProfileFromData
from lasy.profiles.transverse.hermite_gaussian_profile import HermiteGaussianTransverseProfile

from lasy.laser import Laser
import numpy as np


def hermite_gauss_decomposition(laserProfile,nMax=12,mMax=12):
    """
    Decomposes a `lasy` laser profile into a set of hermite-gaussian
    modes. 

    The function takes either an instance of `TransverseProfile` or an 
    instance of `Laser` (that is, either a transverse profile or the 
    full 3D laser profile defined on a grid). In the case that an 
    instance of `Laser` is passed then the intensity of this profile 
    is projected onto an x-y plane for the decomposition.

    Parameters
    ----------
    ???


    """

    # Check if the provided laserProfile is a full laser profile or a
    # transverse profile.

    if isinstance(laserProfile,TransverseProfile):
        
        # Get the intensity, sensible spatial bounds for the profile and a
        # sensible estimate of w0

        lo = [None,None]
        hi = [None,None]
        if isinstance(laserProfile,TransverseProfileFromData):
            lo[0] = laserProfile.field_interp.grid[0].min() + laserProfile.x_offset
            lo[1] = laserProfile.field_interp.grid[1].min() + laserProfile.x_offset
            hi[0] = laserProfile.field_interp.grid[0].max() + laserProfile.y_offset 
            hi[1] = laserProfile.field_interp.grid[1].max() + laserProfile.y_offset
        else:
            lo[0] = -laserProfile.w0*5 + laserProfile.x_offset
            lo[1] = -laserProfile.w0*5 + laserProfile.x_offset
            hi[0] = laserProfile.w0*5 + laserProfile.x_offset
            hi[1] = laserProfile.w0*5 + laserProfile.x_offset

    elif isinstance(laserProfile,Laser):
        # Create a 2D image from the laser data by summing along the 
        # time axis
        intenProf = np.sum(np.abs(laserProfile.field.field)**2,axis=0)

        # Normalise it
        intenProf /= np.sum(intenProf)

    else:
        raise ValueError(
            'Provided profile is not an instance of TransverseProfile or Profile')


    # Next we need to create
    for i in range(nMax):
        for j in range(mMax):
            HGMode = HermiteGaussianTransverseProfile(w0,i,j)
            HGModeInten = HGMode.evaluate()
            coef = np.sum(profile*self.evaluate(X,Y)) # modalDecomposition
            if math.isnan(coef): # due to a numerical issue the coefficient might be nan and will be excluded. Should not happen before mode 63
                coef = 0
            coefficients[(i,j)] = coef