.. LASY documentation master file, created by
   sphinx-quickstart on Thu Dec  8 22:16:12 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

LASY Documentation
================================

LASY (LAser SYmple manipulator) code for manipulating laser pulses. It's primary function is to prepare, in a standard way,
laser pulses as inputs for a variety laser codes related to laser plasma acceleration although its functionality extends
beyond this. It has the ability to define both the transerse and longitudinaly profiles of a laser pulse
and then propagate that pulse over a user defined distance. The library can also incorporate experimentally generated data
such that simulations may be performed with realistic laser pulse inputs. 


This documentation needs fleshing out


.. toctree::
   :caption: Getting Started
   :maxdepth: 1
   :hidden:

   Installation/localMachineInstallation

.. toctree::
   :caption: Overview of the Code
   :maxdepth: 1
   :hidden:

   overview/motivation.rst
   overview/data_standards.rst
   overview/codes_supporting_lasy.rst
   overview/laser_initialization.rst
   overview/laser_propagation.rst

.. toctree::
   :caption: API Reference
   :maxdepth: 1
   :hidden:

   api_reference/laser
   api_reference/utils


.. toctree::
   :caption: Examples
   :maxdepth: 1
   :hidden:

   examples/ex1_initGaussPulse.rst 


