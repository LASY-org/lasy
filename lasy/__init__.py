__version__ = "0.4.0"

from .backend import use_cupy
if use_cupy:
    print('Lasy is using Cupy as a backend.')
else:
    print('Lasy is using Numpy as a backend.')