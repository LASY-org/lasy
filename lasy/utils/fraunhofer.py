import numpy as np
from numpy.fft import ifft,ifftshift,fft,fftshift

def pad_array(input_array,N_pad,axis=0,pad_value=0):
    """
    Pad array along a given axis

    This is used for increasing the spatial resolution for far field calculations

    Parameters
    ----------
    input_array : numpy array
        array to be padded
    
    N_pad : int
        size of array after padding

    axis  : int (optional)
        Axis to be padded (default = 0)
    
    pad_value  : float (optional)
        padding value to append to array when padding

    Returns
    -------
    padded_array: numpy array
        array after padding

    """
    n_dim = len(np.shape(input_array))
    N_input = np.shape(input_array)[axis]
    n = N_pad-N_input
    p_list = [[0,0]]*n_dim
    p_list[axis]= [int(np.ceil(n/2)),int(np.floor(n/2))]

    if n_dim==1: 
        padded_array = np.pad(input_array,p_list[0],'constant',constant_values=pad_value)
    else:
        padded_array = np.pad(input_array,p_list,'constant',constant_values=pad_value)

    return padded_array


def unpad_array(padded_array,N_out,axis=0):
    """
    Remove padding of array along a given axis

    This is used for decreasing the grid size after far field calculations

    Parameters
    ----------
    padded_array : numpy array
        array to reduced in size
    
    N_out : int
        size of array after unpadding

    axis  : int (optional)
        Axis to be unpadded (default = 0)
    
    Returns
    -------
    output_array: numpy array 
        array after unpadding with size N_out in given axis

    """
    N_pad = np.shape(padded_array)[axis]
    # obtain indices of central N_out values
    unpad_ind = np.arange(N_out)+int(N_pad/2-N_out/2)
    output_array = padded_array.take(unpad_ind,axis=axis)
    return output_array

def calc_fft_pad(E_x,axis=0,N_pad=None,unpad_result=True,inverse=False):
    """
    Calculate the fourier transform of a field (or its inverse) with optional padding

    This is used for far field calculations

    Parameters
    ----------
    E_x : numpy array of complex
        Field array to be fourier transformed
    
    N_pad : int (optional)
        size of padded array if padding is to be used in fourier transform

    unpad_result : boolean (optional)
        if True return array size to original after fourier transform
    
    inverse : boolean (optional)
        if true use ifft rather than fft
    
    Returns
    -------
    E_u: numpy array of complex
        array after padding, fourier transforming and unpadding along transverse direction

    """
    N_input = np.shape(E_x)[axis]
    if N_pad is not None:
        E_x = pad_array(E_x,N_pad,axis=axis)

    if inverse:
        fft_func = ifft
    else:
        fft_func = fft

    E_u = fftshift(fft_func(ifftshift(E_x,axes=axis),axis=axis),axes=axis)

    if unpad_result:
        E_u = unpad_array(E_u,N_input,axis=axis)
    return E_u


def calc_farfield_fraunhofer(E_x,d_fx=1,axis=0,N_pad=None,unpad_result=True):
    """
    Calculate the focus of a laser field using Fraunhofer diffraction

    Parameters
    ----------
    E_x : numpy array of complex
        Field array at near field
    
    d_fx : float (optional)
        step size in angular frequency of (original unpadded) near field (i.e. d_fx = d_x/(f*lambda0) )

    N_pad : int (optional)
         size of padded array if padding is to be used in fourier transform
         Increase to improve spatial resolution of result
    
    unpad_result : boolean (optional)
        if True return array size to original after focus calculation
    
    
    Returns
    -------
    E_u: numpy array of complex
        far field array

    """
    if N_pad is None:
        N_x = np.shape(E_x)[axis]
    else:
        N_x = N_pad
    E_u = calc_fft_pad(E_x,axis=axis,N_pad=N_pad,unpad_result=unpad_result,inverse=True)*d_fx*N_x
    return E_u


def calc_fraunhofer_axis(N,d_fx,N_pad=None,unpad_result=True):
    """
    Returns the spatial axis for a corresponding far field.

    This should be used with calc_farfield_fraunhofer to give the corresponding spatial axis


    Parameters
    ----------
    N : int
        Original size of near field
    
    d_fx : float (optional)
        step size in angular frequency of (original unpadded) near field (i.e. d_fx = d_x/(f*lambda0) )
        (should value used in calc_farfield_fraunhofer)

    N_pad : int (optional)
        size of padded array if padding was used in fourier transform 
        (should value used in calc_farfield_fraunhofer)
    
    unpad_result : boolean (optional)
        if True return array size to original (should value used in calc_farfield_fraunhofer)
    
    
    Returns
    -------
    E_u: numpy array of complex
        far field array

    """
    if N_pad is None:
        N_pad = N
    du = 1/((N_pad)*d_fx)
    u = (np.arange(N_pad)-N_pad/2)*du
    if unpad_result:
        u = unpad_array(u,N)
    return u