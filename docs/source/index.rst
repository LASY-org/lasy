.. LASY documentation master file, created by
   sphinx-quickstart on Thu Jan  5 15:49:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

LASY |release| Documentation
============================

``lasy`` (LAser manipulations made eaSY) is a Python library that facilitates the initialization of complex laser pulses, in simulations of laser-plasma interactions.

More specifically, ``lasy`` offers many ways to define complex laser pulses (e.g. from commonly-known analytical formulas, from experimental measurements, etc.) and offers pre-processing functionalities (e.g. propagation, re-normalization, geometry conversion).
The laser field is then exported in a standardized file, that can be read by external simulation codes.

The code is open-source and hosted on `github <https://github.com/LASY-org/lasy>`__. Contributions are welcome!

.. grid:: 1 1 2 2
    :gutter: 2

    .. grid-item-card:: Getting Started
        :text-align: center

        New to ``lasy``? Check this out for installation instructions and a first example.

        +++

        .. button-ref:: user_guide/index
                :expand:
                :color: primary
                :click-parent:

                More Information

    .. grid-item-card:: Overview of the Code
        :text-align: center

        An overview of the key concepts and functionality of the code.

        +++

        .. button-ref:: code_overview/index
                :expand:
                :color: primary
                :click-parent:

                Get an overview of the code

    .. grid-item-card:: API Reference
        :text-align: center

        Get into the nuts and bolts of the ``lasy`` API with the documentation here.

        +++

        .. button-ref:: api/index
                :expand:
                :color: primary
                :click-parent:

                Take a look at the API Reference

    .. grid-item-card:: Tutorials
        :text-align: center

        Some step-by-step guides to using the code and some common examples which you might find useful.

        +++

        .. button-ref:: tutorials/index
                :expand:
                :color: primary
                :click-parent:

                Show me some Tutorials

.. toctree::
   :hidden:
   :maxdepth: 4

   user_guide/index
   code_overview/index
   api/index
   tutorials/index
