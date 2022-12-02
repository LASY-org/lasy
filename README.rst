lasy
####

Documentation
-------------

LASY manipulates laser pulses, and operates on the laser envelope. The definition used is:

.. math::

   1+1=2
   E_x (x,y,t) = \Re ( \mathcal{E} e^{-i\omega_0t}p_x)
 
Style conventions
-----------------

- Docstrings are written using the Numpy style.
- For each significant contribution, using pull requests is encouraged: the description helps to explain the code and open dicussion.

Test
----

.. code-block:: bash

   python setup.py install
   python examples/test.py

