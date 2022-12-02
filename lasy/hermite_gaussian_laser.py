from .laser import laser

class HermiteGaussianLaser(Laser):
    """
    Derived class for an analytic profile of high-order Gaussian
    laser pulses in xyz coordinates.
    """

    def __init__(dim, lo, hi, wavelength, pol, emax, tau, w0):
        """
        Hermite Gaussian laser constructor

        We follow Siegman's definition of Hermite-Gauss modes. 
        We use the representation of Hermite-Gauss modes with a 
        real argument in the Hermite polynomial and complex argument 
        in the exponential as these form an orthonormal set and can 
        be used as a basis set to expand arbitrary higher order 
        paraxial beams. 

        These modes may be defined as
        $$
        E_n (x,z) = 
        \left ( \frac{2}{\pi} \right)^{1/4} 
        \sqrt{
            \frac{
                \exp \left [  - i (2 n + 1 ) ( \psi(z) - \psi_0)\right]}
                {2^n n! w(z)}} 
        \times H_n\left ( \frac{\sqrt{2}x }{w(z)}\right ) 
        \exp\left [ -i \frac{kx^2}{2R(z)} - \frac{x^2}{w^2(z)}\right],
        $$
        with
        $$
        \tan \psi(z) \equiv \frac{\pi w^2(z)}{R(z)\lambda},
        $$
        and 
        $$
        \psi_0 \equiv \psi(z_0).
        $$
        Additionally, we calculate $R(z)$ from the complex radius $\tilde{q}(z)$ , where
        $$
        \frac{1}{\tilde{q}(z)} \equiv \frac{1}{R(z)} - i \frac{\lambda}{\pi w^2(z)},
        $$
        and $\tilde{q(z)}$ evolves as
        $$
        \tilde{q}(z) = \tilde{q_0} + z - z_0.
        $$
        
        """
        # TODO
    
