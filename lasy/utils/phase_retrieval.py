import copy

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
        
        self.laser1 = copy.deepcopy(self.lasers[0])
        self.laser2 = copy.deepcopy(self.lasers[1])
        
        self.amp1 = np.abs(self.laser1.grid.get_temporal_field())
        self.amp2 = np.abs(self.laser2.grid.get_temporal_field())
        
        self.x = self.laser2.grid.axes[0]
        self.y = self.laser2.grid.axes[1]
        self.z = self.positions
        
        
    def initialise_spotsizes(self,spotsizes=None):
        """
        Initialise the spot size of the modes at focus in each transverse direction
        
        Parameters
        ----------
        spotSizes: tuple of floats (meters)
            The size of the mode along the x and y axes at focus. spotSize[0] corresponds to 
            the y-axis while spotSize[1] corresponds to the x-axis.  If none is provided then
            the spot sizes will be initialised based on a D4sigma fit to the
            focus.
            
        """

        # The spotsize is calculated from the fluence, profile is integrated wrt temporal axis
        if spotsizes is None:
            w0x, w0y = estimate_best_HG_waist(
                self.x, self.y, np.sum(self.amp2, axis=-1), self.laser2.profile.lambda0
            ) # Estimate spot size in the focal plane
            spotsizes = (w0x, w0y)

        self.spotsizes = spotsizes

    
    def initialise_modes(self):
        

        # Initialise random phase if no known phase is passed
        if self.initial_phase is None:
            self.phase1 = np.pi * np.random.uniform(-1, 1, np.shape(self.amp1)) # Initial guess of phase
        else:
            self.phase1 = self.initial_phase

        # Construct initial guess of field
        self.laser1.grid.set_temporal_field(self.amp1*np.exp(1j*self.phase1))
        
        # Find estimate of the decomposition of the initial electric field
        self.modes = hermite_gauss_decomposition(self.laser2, self.spotsizes[0], self.spotsizes[1], self.m_max, self.n_max)
    
    def retrieve_phase(self):
    
        def breakout(i):
            return i < self.max_iter
        cond = 0

        errors = []
    
        i = 0
        while breakout(cond):
            self.laser1.grid.set_temporal_field(self.amp1 * np.exp(1j * self.phase1))

            
            # Calculate the decomposition and waist of the laser pulse
            self.modes = hermite_gauss_decomposition(self.laser1, self.spotsizes[0], self.spotsizes[1], self.m_max, self.n_max)

            self.laser1.propagate(self.dz, verbose=False)
    
            self.phase2 = np.angle(self.laser1.grid.get_temporal_field())
            self.laser2.grid.set_temporal_field(self.amp2 * np.exp(1j * self.phase2))
            self.laser2.propagate(-self.dz, verbose=False)
    
            phase1 = np.angle(self.laser2.grid.get_temporal_field())
            
            amp_error_summed = np.sum(np.abs(np.abs(self.laser2.grid.get_temporal_field())-self.amp1)) / np.sum(self.amp1)
            
            i += 1
            cond += 1
            if self.verbose:
                print(
                    "Iteration %i : Amplitude Error (summed) = %.2e"
                    % (i, amp_error_summed)
                )

            errors.append(amp_error_summed)
    
        return phase1, phase2, np.array(errors)