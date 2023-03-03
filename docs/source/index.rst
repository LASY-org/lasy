.. LASY documentation master file, created by
   sphinx-quickstart on Thu Jan  5 15:49:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

LASY |release| Documentation
============================

.. warning::
        Warning: This library is currently in development, and it is, at this stage, only meant to be used/tested by developers.
        We plan on releasing the first version for general users (i.e. beta version) by summer 2023.



``lasy`` (LAser SYmple manipulator) is a Python library that facilitates the initialization of complex laser pulses, in simulations of laser-plasma interactions.

More specifically, ``lasy`` offers many ways to define complex laser pulses (e.g. from commonly-known analytical formulas, from experimental measurements, etc.) and offers pre-processing functionalities (e.g. propagation, re-normalization, geometry conversion).
The laser field is then exported in a standardized file, that can be read by external simulation codes.

The code is open-source and hosted on `github <https://github.com/LASY-org/lasy>`__. Contributions are welcome!

.. panels::
    :card: + intro-card text-center
    :column: col-lg-6 col-md-6 col-sm-6 col-xs-12 p-2 d-flex

    ---

    **Getting Started**
    ^^^^^^^^^^^^^^

    New to ``lasy``? Check this out for installation instructions and a first example.

    +++

    .. link-button:: user_guide/index
            :type: ref
            :text: More Information
            :classes: btn-outline-primary btn-block stretched-link

    ---

    **Overview of the Code**
    ^^^^^^^^^^^^^^^^^^^^^^^^

    An overview of the key concepts and functionality of the code.

    +++

    .. link-button:: code_overview/index
            :type: ref
            :text: Get an overview of the code
            :classes: btn-outline-primary btn-block stretched-link

    ---

    **API Reference**
    ^^^^^^^^^^^^^^^^^

    Get into the nuts and bolts of the ``lasy`` API with the documentation here.

    +++

    .. link-button:: api/index
            :type: ref
            :text: Take a look at the API Reference
            :classes: btn-outline-primary btn-block stretched-link

    ---

    **Tutorials**
    ^^^^^^^^^^^^^

    Some step-by-step guides to using the code and some common examples which you might find useful.

    +++

    .. link-button:: tutorials/index
            :type: ref
            :text: Show me some tutorials
            :classes: btn-outline-primary btn-block stretched-link

.. toctree::
   :hidden:
   :maxdepth: 4

   user_guide/index
   code_overview/index
   api/index
   tutorials/index
