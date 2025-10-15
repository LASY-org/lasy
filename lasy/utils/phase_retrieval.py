import copy
import time

import numpy as np
import matplotlib.pyplot as plt
from lasy.optical_elements.parabolic_mirror import ParabolicMirror
from lasy.utils.mode_decomposition import (
    estimate_best_HG_waist,
    hermite_gauss_composition,
    hermite_gauss_decomposition,
)

class GerchbergSaxton():
    """
    An implementation of the Gerchberg Saxton Algorithm with Modal Decomposition (GSA-MD)
    as described by I. Moulanier et al., Jour. Opt. Soc. Am. B 40, 9 (2023). DOI 10.1364/JOSAB.489884
    
    """
    
    def __init__(self, lasers, positions, m_max=10, n_max=10, max_iter=100, initial_phase=None, verbose=False):
        """
        Parameters
        ----------

    

        """
        self.lasers = lasers
        self.positions = positions
        self.m_max = m_max
        self.n_max = n_max
        self.max_iter = max_iter
        self.initial_phase = initial_phase
        self.verbose = verbose   

        self.initialise_lasers()
        self.initialise_spotsizes()
        self.initialise_modes()

    
    def initialise_lasers(self):
        """
        Initialise the laser amplitudes and axes from the laser objects
        
        Parameters
        ----------
            
        """
        
        self.amps = []
        for laser in self.lasers:
            self.amps.append(np.abs(laser.grid.get_temporal_field()))
            print(np.shape(np.abs(laser.grid.get_temporal_field())))
        
        self.x = self.lasers[0].grid.axes[0]
        self.y = self.lasers[0].grid.axes[1]
        self.z = self.positions
        self.z0idx = np.argmin(np.abs(self.z)) # Position of plane closest to the focus
        
        
    def initialise_spotsizes(self,spotsizes=None):
        """
        Initialise the spot size of the modes closest to the focus in each transverse direction
        
        Parameters
        ----------
        spotSizes: tuple of floats (meters)
            The size of the mode along the x and y axes at focus. spotSize[0] corresponds to 
            the y-axis while spotSize[1] corresponds to the x-axis.  If none is provided then
            the spot sizes will be initialised based on a D4sigma fit to the
            focus.
            
        """

        # The spotsize is calculated from the fluence, profile is integrated wrt temporal axis
        # Pick the plane closest to the focus
        if spotsizes is None:
            w0x, w0y = estimate_best_HG_waist(
                self.x, self.y, np.sum(self.amps[self.z0idx], axis=-1), self.lasers[self.z0idx].profile.lambda0,
            ) # Estimate spot size in the focal plane
            spotsizes = (w0x, w0y)

        self.spotsizes = spotsizes

    
    def initialise_modes(self):
        """
        Initialise the phase and mode coefficients closest to the focus in each transverse direction
        
        Parameters
        ----------
            
        """
        

        # Initialise random phase if no known phase is passed
        if self.initial_phase is None:
            self.phase = np.pi * np.random.uniform(-1, 1, np.shape(self.amps[0])) # Initial guess of phase
        else:
            self.phase = self.initial_phase

        # Construct initial guess of field
        self.lasers[self.z0idx].grid.set_temporal_field(self.amps[self.z0idx]*np.exp(1j*self.phase))
        
        # Find estimate of the decomposition of the initial electric field
        self.modes = hermite_gauss_decomposition(self.lasers[self.z0idx], self.spotsizes[0], self.spotsizes[1], self.m_max, self.n_max, z_foc=self.z[self.z0idx])

    
    def retrieve_phase(self):
        """
        Perform the phase retrieval over the given planes
        
        Parameters
        ----------
            
        """
    
        # Make an array to alternate from the ends of array, working towards center
        idx = np.empty(len(self.z), dtype=int)
        half_ceil = (len(self.z) + 1) // 2  # Ceiling division for the first half
        idx[::2] = np.arange(half_ceil)
        half_floor = len(self.z) // 2  # Floor division for the second half
        idx[1::2] = np.arange(len(self.z) - 1, half_ceil - 1, -1)
        
        mode_power = sum(abs(value) ** 2 for value in self.modes.values())
        chi2 = np.zeros(self.max_iter)
        chi2Grad = np.zeros(self.max_iter)
    
        i = 0
        for i in range(self.max_iter):
            if self.verbose: print("GSA Iteration: %i" %i)
            for k in idx:
                if self.verbose: print("    Image: %i of %i" %(k+1,len(self.z)))

                # Step 2
                t0 = time.time()
                
                hermite_gauss_composition(self.lasers[k], self.spotsizes[0], self.spotsizes[1], self.modes, z_foc=self.z[k])
                                                   
                t1 = time.time()
                if self.verbose: print("        Reconstruction: %.3f s" %(t1-t0))
                
                # Step 3
                phi = np.angle(self.lasers[k].grid.get_temporal_field())

                # Step 4
                laser_new = self.amps[k] * np.exp(1j*phi)

                # Step 5
                delta = (self.amps[k] - np.abs(self.lasers[k].grid.get_temporal_field()))/np.max(self.amps[k])
                laser_new *= np.exp(delta)

                self.lasers[k].grid.set_temporal_field(laser_new)

                # Step 6
                t0 = time.time()
                
                new_modes = hermite_gauss_decomposition(self.lasers[k], self.spotsizes[0], self.spotsizes[1], self.m_max, self.n_max, z_foc=self.z[k])
                
                t1 = time.time()
                if self.verbose: print("        Decomposition:  %.3f s" %(t1-t0))
                
                
                for key in new_modes:
                    new_modes[key] *= np.sqrt(mode_power/sum(abs(value) ** 2 for value in new_modes.values()))
                    self.modes[key] = (self.modes[key] + new_modes[key])/2.

                for key in self.modes:
                    self.modes[key] *= np.sqrt(mode_power/sum(abs(value) ** 2 for value in self.modes.values()))
                    
                # calculate the fluence error
                chi2[i] = np.sqrt(np.sum((np.abs(laser_new)  - self.amps[k]) **2))/np.sum(self.amps[k])/len(self.z)

            if i>0:
                chi2Grad[i] = np.abs(chi2[i] - chi2[i-1])/(chi2[i-1]+1e-8) # Add a small amount to prevent inf
                if self.verbose:
                    print("chi2     = %.7e " %(chi2[i]))
                    print("chi2Grad = %.7f %%" %(chi2Grad[i]*100))

            i+=1
        
        return self.lasers, chi2, chi2Grad