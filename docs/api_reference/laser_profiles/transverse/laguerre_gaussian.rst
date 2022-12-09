Laguerre-Gaussian Laser Pulse
==============================

Used to define a Laguerre-Gaussian transverse laser profile. These are a family
of 

.. math::
    E_u(r,\theta,t) = Re\left[ E_0\, r^{|m|}e^{-im\theta} \,
    L_p^{|m|}\left( \frac{2 r^2 }{w_0^2}\right )\,
    \exp\left( -\frac{r^2}{w_0^2}
    - \frac{(t-t_{peak})^2}{\tau^2} -i\omega_0(t-t_{peak})
    + i\phi_{cep}\right) \times p_u \right]

where :math:`u` is either :math:`x = r \cos{\theta}` or 
:math:`y = r \sin{\theta}`, :math:`L_p^{|m|}` is the
Generalised Laguerre polynomial of radial order :math:`p` and
azimuthal order :math:`|m|`, :math:`p_u` is the polarization
vector, :math:`Re` represent the real part, and :math:`r` is the radial
coordinate (orthogonal to the propagation direction) and :math:`\theta`
is the azmiuthal coordinate and :math:`t` is time. The other parameters
in this formula are defined below.