from .single_fft_propagator import SingleFFTPropagator

from laser_utils import get_w0
c = 2.998e8


class CollinsSFFTPropagator(SingleFFTPropagator):
    def __init__(self):
        super().__init__()
        #self.update()

        abcd = self.abcd

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


    def add_output_grid(self, grid_in, N_points, magnification=None, f0=None):
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
        
        # AT THE MOMENT THIS ASSUMES SYMMETRIC XY GRIDS

        axes = grid_in.axes # xyt
        
        
        self.w0 = self.laser.get_w0(self.grid, self.dim)
        lambda_0 = 2. * np.pi * c / self.omega_0
        
        N_points = 100
        padFactor = int(np.ceil(magnification * (2. * w_0 / f_0) * L0_width / lambda_0 / N_points)) // 2 * 2 + 1 # Must be odd
        print("Padding factor: ", padFactor)
        
        print("Number of spatial/transverse gridpoints: %0.0f" % (N_points))
        r0_step = np.abs(np.max(grid_in[0])-np.min(grid_in[0])) / (N_points*padFactor)  # Note: D gridpoints means D-1 intervals
        
        x = fftshift(fftfreq(N_points*padFactor, r0_step) * lambda_0 * f_0) / padFactor
        y = fftshift(
            fftfreq(N_points*padFactor, r0_step) * lambda_0 * f_0
        ) / padFactor
        
        # Simulation output meshgrid
        X, Y = np.meshgrid(
            x,
            y,
            indexing="ij",
        )
        R = np.sqrt(X**2 + Y**2)
        
        # Calculate unpadded output grids
        region_idx = np.array([[np.shape(R)[0]//2-W//2,
                      np.shape(R)[0]//2+W//2],
                      [np.shape(R)[1]//2-W//2,
                      np.shape(R)[1]//2+W//2]]) # Indexing of the focal region of interest
        x_grid = x[region_idx[0,0]:region_idx[0,1]]
        y_grid = y[region_idx[1,0]:region_idx[1,1]]
        dx = np.abs(x_grid[1] - x_grid[0]) # Calculate resolutions
        dy = np.abs(y_grid[1] - y_grid[0])

        self.grid_in = R
        return R, x_grid, y_grid
    

    def propagate(self, distance, grid_in=None, grid_out=None, abcd=None):
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
        # ASSUMES INPUT GRID IS MONOCHROMATIC MESHGRID OF R_in
        
        self.update()

        if grid_out = None:
            grid_out = self.grid_out
        else:
            pass
            
        if abcd = None:
            abcd = self.abcd
        else:
            pass

        try:
            A = matrix[0][0]
            B = matrix[0][1]
            C = matrix[1][0]
            D = matrix[1][1]
        except:
            print("Missing the ray matrix for the optical system.")

        profile_in = self.profile_in

        padNumberx = int(np.shape(grid_out)[0]/np.shape(grid_in)[0])//2 * np.shape(grid_in)[0]
        padNumbery = int(np.shape(grid_out)[1]/np.shape(grid_in)[1])//2 * np.shape(grid_in)[1]
    
        propagator = np.exp(1j * omega_0 / (2 * c) * (A / B) * grid_in**2)
    
        profile_in = np.pad(profile_in, [(padNumberx, padNumbery), (padNumberx, padNumbery)], mode='constant')
        propagator = np.pad(propagator, [(padNumberx, padNumbery), (padNumberx, padNumbery)], mode='constant')
        print("Shape of padded profile: ",np.shape(profile_in))
        
        profile_out = fftshift(
            ifft2(
                ifftshift(profile_in * propagator, axes=(0, 1)),
                axes=(0, 1),
            ),
            axes=(0, 1),
        ) * np.sqrt(np.shape(grid_out)[0] * np.shape(grid_out)[1])
        profile_out = (
            profile_out
            * np.exp(1j * omega_0 / (2 * c) * (D / B) * grid_out**2)
            * omega_0
            / (2j * np.pi * c * B)
            / np.abs(omega_0 / (2j * np.pi * c * B))
        )
        
        profile_out = profile_out[region_idx[0,0]:region_idx[0,1],region_idx[1,0]:region_idx[1,1]]  # Select ROI
        return profile_out