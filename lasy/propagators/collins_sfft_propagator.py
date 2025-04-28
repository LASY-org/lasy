from .single_fft_propagator import SingleFFTPropagator

from lasy.utils.laser_utils import get_w0
c = 2.998e8


class CollinsSFFTPropagator(SingleFFTPropagator):
    def __init__(self):
        super().__init__()
        #self.update()

        self.abcd = np.array([[1, 0],[0, 1]])

        return

    def add_vacuum(self, distance):
        vacuum = np.array([[1, distance],[0, 1]])
        self.abcd = np.matmul(vacuum, self.abcd)
        return self.abcd
        

    def add_lens(self, focal_length):
        self.f0 = focal_length
        lens = np.array([[1, 0],[- 1./focal_length, 1]])
        self.abcd = np.matmul(lens, self.abcd)
        return self.abcd


    def add_output_grid(self, grid, N_points=100, magnification=1.0, f0=None):
        """
        Function to calculate the output grids 
        for a focusing / defocusing beam
    
        Parameters
        ----------
        grid_in : meshgrid (in meter)
            2D meshgrid for the input coordinates
            
        N_points : int
            The desired number of gridpoints in the output plane
            
        magnification : float (factor)
            The desired magnification of the focus 
            A parameter that can be tuned for the 
            output resolution
    
        f0 : float (in meter)
            The focal length of the lens
    
        """        
        
        # AT THE MOMENT THIS ASSUMES SYMMETRIC CARTESIAN GRIDS
        axes = grid.axes
        x = axes[0]
        y = axes[1]
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        w0 = get_w0(grid, self.dim)
        f0 = self.f0
        lambda_0 = 2. * np.pi * c / self.omega_0
        
        padFactor = int(np.ceil(magnification * (2. * w0 / f0) * L0_width / lambda_0 / N_points)) // 2 * 2 + 1 # Must be odd
        print("Padding factor: ", padFactor)
        
        print("Number of spatial/transverse gridpoints: %0.0f" % (N_points))
        r0_step = np.abs(np.max(x)-np.min(x)) / (N_points*padFactor)  # Note: D gridpoints means D-1 intervals
        
        x = fftshift(fftfreq(N_points*padFactor, r0_step) * lambda_0 * f0) / padFactor
        y = fftshift(
            fftfreq(N_points*padFactor, r0_step) * lambda_0 * f0
        ) / padFactor
        
        # Simulation output meshgrid
        X, Y = np.meshgrid(
            x,
            y,
            indexing="ij",
        )
        R = np.sqrt(X**2 + Y**2)
        
        # Calculate unpadded output grids
        region_idx = np.array([[np.shape(R)[0]//2-N_points//2,
                      np.shape(R)[0]//2+N_points//2],
                      [np.shape(R)[1]//2-N_points//2,
                      np.shape(R)[1]//2+N_points//2]]) # Indexing of the focal region of interest
        return [x,y], region_idx
    

    def propagate(self, grid, grid_out=None, distance=None, abcd=None):
        """
        Function to calculate an output field from 
        input field and optical ray matrix of the system
    
        Parameters
        ----------
        profile : array
            The input field profile with arbitrary intensity and phase
    
        grid_in : meshgrid (in meter)
            2D meshgrid for the input coordinates
    
        grid_out : meshgrid (in meter)
            2D meshgrid for the output coordinates
    
        abcd : array
            The ray matrix of the optical system
    
        omega_0 : float (in meter)
            The wavelength of the electric field
    
        """
        axes = grid.axes
        omega0 = self.omega0

        # Get the spectral field and axes from the input grid
        spectral_field, spectral_axes = grid.get_spectral_field()
        
        if grid_out = None:
            axes_out, region_idx = add_output_grid(grid) # Call routine to determine output grid
        else:
            axes_out = self.grid_out.axes # Use user-specified grid
            
        if abcd = None: # Update ABCD matrix if passed as variable
            abcd = self.abcd
        else:
            pass

        try:
            A = abcd[0][0]
            B = abcd[0][1]
            C = abcd[1][0]
            D = abcd[1][1]
        except:
            print("Missing the ray matrix for the optical system.")
            

        if self.dim == "rt":
            print("Collins SFFT propagator in rt")
            profile_out = None

        elif self.dim == "xyt":
            print("Collins SFFT propagator in xyt")
            axes = grid.axes
            x0 = axes[0]
            y0 = axes[1]
            dx0 = x0[1] - x0[0]
            dy0 = y0[1] - y0[0]

            x = axes_out[0]
            y = axes_out[1]
            dx = x[1] - x[0]
            dy = y[1] - y[0]

            X0, Y0, OM = np.meshgrid(y0[1], x0[0], spectral_axes + omega0)
            X, Y, OM = np.meshgrid(y[1],  x[0],  spectral_axes + omega0)
            K = OM / c
            WAVELENGTH = 2 * np.pi * c / OM
            
            R0 = np.sqrt(X0**2 + Y0**2)
            R = np.sqrt(X**2 + Y**2)
            
            profile_in = spectral_field
    
            padNumberx = int(np.shape(axes_out)[0]/np.shape(axes)[0])//2 * np.shape(axes)[0]
            padNumbery = int(np.shape(axes_out)[1]/np.shape(axes)[1])//2 * np.shape(axes)[1]
    
            propagator = np.exp(1j * OM / (2 * c) * (A / B) * R0**2)
            
            profile = np.pad(profile, [(padNumberx, padNumbery), (padNumberx, padNumbery)], mode='constant')
            propagator = np.pad(propagator, [(padNumberx, padNumbery), (padNumberx, padNumbery)], mode='constant')
            print("Shape of padded profile: ",np.shape(profile))
            
            profile_out = fftshift(
                ifft2(
                    ifftshift(profile * propagator, axes=(0, 1)),
                    axes=(0, 1),
                ),
                axes=(0, 1),
            ) * np.sqrt(np.shape(R)[0] * np.shape(R)[1])
            profile_out = (
                profile_out
                * np.exp(1j * OM / (2 * c) * (D / B) * R**2)
                * omega_0
                / (2j * np.pi * c * B)
                / np.abs(OM / (2j * np.pi * c * B))
            )
    
            profile_out = profile_out[region_idx[0,0]:region_idx[0,1],region_idx[1,0]:region_idx[1,1]]  # Select ROI
            x = x[region_idx[0,0]:region_idx[0,1]]
            y = y[region_idx[1,0]:region_idx[1,1]]
            
        # Update grid here to be the new output grid and spectral field to be new spectral field
        grid_new = Grid(self.dim, [np.min(x),np.min(y)], [np.max(x),np.max(y)], [len(x),len(y)])
        laser.grid = grid_new
        grid.set_spectral_field(profile_out)
        
        return profile_out