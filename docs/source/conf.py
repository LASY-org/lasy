# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from datetime import date

from lasy import __version__

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "LASY"
copyright = "%s, LASY-org" % date.today().year
author = "LASY-org"

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",
    "nbsphinx",
]

# Numpydoc settings
numpydoc_show_class_members = False
numpydoc_use_plots = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Logo
html_logo = "https://user-images.githubusercontent.com/27694869/211026764-b55a7406-3e7c-44ab-b314-a08c75aca46e.png"

# Theme options
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/LASY-org/lasy",
            "icon": "fa-brands fa-github",
        },
    ],
    "navigation_with_keys": False,
}

# Prevent panels extension from modifying page style.
panels_add_bootstrap_css = False

# Document __init__ class methods
autoclass_content = "both"

# Configuration for generating tutorials.
from sphinx_gallery.sorting import FileNameSortKey

sphinx_gallery_conf = {
    "examples_dirs": "../../tutorials",
    "gallery_dirs": "tutorials",
    "filename_pattern": ".",
    "within_subsection_order": FileNameSortKey,
}

# ------------------------------------------------------------------------------
# Matplotlib plot_directive options
# ------------------------------------------------------------------------------

plot_include_source = True
plot_formats = [("png", 96)]
plot_html_show_formats = False
plot_html_show_source_link = False

import math

phi = (math.sqrt(5) + 1) / 2

font_size = 13 * 72 / 96.0  # 13 px

plot_rcparams = {
    "font.size": font_size,
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": font_size,
    "figure.figsize": (3 * phi, 3),
    "figure.subplot.bottom": 0.2,
    "figure.subplot.left": 0.2,
    "figure.subplot.right": 0.9,
    "figure.subplot.top": 0.85,
    "figure.subplot.wspace": 0.4,
    "text.usetex": False,
}
