import numpy as np
import math

from lasy.profiles.transverse.hermite_gaussian_profile import (
    HermiteGaussianTransverseProfile
)
from lasy.profiles.transverse.laguerre_gaussian_profile import (
    LaguerreGaussianTransverseProfile
)
from lasy.profiles.transverse.transverse_profile import TransverseProfile
from lasy.utils.exp_data_utils import find_d4sigma


def getHGMode(grid_in, w0x, w0y, i, j):

    X, Y, T = np.meshgrid(grid_in.grid.axes[0], grid_in.grid.axes[1], grid_in.grid.axes[2], indexing='ij')
    
    dx = np.mean(np.diff(grid_in.grid.axes[0]))
    dy = np.mean(np.diff(grid_in.grid.axes[1]))
    
    hg = HermiteGaussianTransverseProfile(w0x, w0y, i, j, grid_in.profile.lambda0).evaluate(X,Y)
    
    coeff = np.sum(grid_in.grid.get_temporal_field()*np.conj(hg))*dx*dy
    
    return coeff
    
def decomposeHG(grid_in, w0x, w0y, Mmax, Nmax, skipAsymmetricModes=False):
    
    X, Y, T = np.meshgrid(grid_in.grid.axes[0], grid_in.grid.axes[1], grid_in.grid.axes[2], indexing='ij')
    field = grid_in.grid.get_temporal_field()
    
    cxy = {}
    for i in range(Mmax):
        for j in range(Nmax):
            if (i != j) and (skipAsymmetricModes):
                cxy[(i,j)] = 0
                continue
            cxy[(i,j)] = getHGMode(grid_in, w0x, w0y, i, j)
        
    return cxy

def reconstructHG(grid_in, w0x, w0y, cxy, skipAsymmetricModes=False):

    X, Y, T = np.meshgrid(grid_in.grid.axes[0], grid_in.grid.axes[1], grid_in.grid.axes[2], indexing='ij')
    field = np.zeros(X.shape, dtype='complex128')

    for nxy in list(cxy):
        if (nxy[0] != nxy[1]) and (skipAsymmetricModes):
            continue
        field += cxy[nxy]*(HermiteGaussianTransverseProfile(w0x, w0y, nxy[0], nxy[1], grid_in.profile.lambda0).evaluate(X,Y))

    grid_in.grid.set_temporal_field(field)



def getLGMode(grid_in, w0, i, j):

    X, Y, T = np.meshgrid(grid_in.grid.axes[0], grid_in.grid.axes[1], grid_in.grid.axes[2], indexing='ij')
    
    dx = np.mean(np.diff(grid_in.grid.axes[0]))
    dy = np.mean(np.diff(grid_in.grid.axes[1]))
    
    lg = LaguerreGaussianTransverseProfile(w0, i, j, grid_in.profile.lambda0).evaluate(X,Y)
    
    coeff = np.sum(grid_in.grid.get_temporal_field()*np.conj(lg))*dx*dy
    
    return coeff
    
def decomposeLG(grid_in, w0, Mmax, Nmax, skipAsymmetricModes=False):
    
    X, Y, T = np.meshgrid(grid_in.grid.axes[0], grid_in.grid.axes[1], grid_in.grid.axes[2], indexing='ij')
    field = grid_in.grid.get_temporal_field()
    
    cxy = {}
    for i in range(Mmax):
        for j in range(Nmax):
            if (i != j) and (skipAsymmetricModes):
                cxy[(i,j)] = 0
                continue
            cxy[(i,j)] = getLGMode(grid_in, w0, i, j)
        
    return cxy

def reconstructLG(grid_in, w0, cxy, skipAsymmetricModes=False):

    X, Y, T = np.meshgrid(grid_in.grid.axes[0], grid_in.grid.axes[1], grid_in.grid.axes[2], indexing='ij')
    field = np.zeros(X.shape, dtype='complex128')

    for nxy in list(cxy):
        if (nxy[0] != nxy[1]) and (skipAsymmetricModes):
            continue
        field += cxy[nxy]*(LaguerreGaussianTransverseProfile(w0, nxy[0], nxy[1], grid_in.profile.lambda0).evaluate(X,Y))

    grid_in.grid.set_temporal_field(field)