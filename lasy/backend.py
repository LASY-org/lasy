try:
    import cupy as xp

    use_cupy = True
except ImportError:
    import numpy as xp

    use_cupy = False
