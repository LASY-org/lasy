try:
    import cupy as xp
    from cupyx.scipy.interpolate import RegularGridInterpolator
    from cupyx.scipy.signal import hilbert, zoom_fft
    from cupyx.scipy.special import j0

    use_cupy = True

    def to_cpu(arr):
        """Convert array from cupy to numpy."""
        if isinstance(arr, xp.ndarray):
            return xp.asnumpy(arr)
        elif isinstance(arr, list):
            return [to_cpu(a) for a in arr]
        elif isinstance(arr, tuple):
            return (to_cpu(a) for a in arr)
        else:
            return arr

    def to_gpu(arr):
        """Convert array from numpy to cupy."""
        if not isinstance(arr, xp.ndarray):
            return xp.asarray(arr)
        else:
            return arr

except ImportError:
    import numpy as xp
    from scipy.interpolate import RegularGridInterpolator
    from scipy.signal import hilbert, zoom_fft
    from scipy.special import j0

    use_cupy = False

    def to_cpu(arr):
        """Convert array from cupy to numpy."""
        return arr

    def to_gpu(arr):
        """Convert array from numpy to cupy."""
        return arr


__all__ = [
    "use_cupy",
    "xp",
    "to_cpu",
    "to_gpu",
    "RegularGridInterpolator",
    "hilbert",
    "zoom_fft",
    "j0",
]
