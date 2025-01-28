import numpy as np
from numpy.fft import ifft,ifftshift,fft,fftshift

def unpad_array(a_pad,N,axis=0):
    N_pad = np.shape(a_pad)[axis]
    unpad_ind = np.arange(N)+int(N_pad/2-N/2)
    a = a_pad.take(unpad_ind,axis=axis)
    return a

def calc_focus_fraunhofer(E_x,dx=1,axis=0,N_pad=None,unpad_result=True):
    if N_pad is None:
        N_x = np.shape(E_x)[axis]
    else:
        N_x = N_pad
    E_u = calc_ifft_pad(E_x,axis=axis,N_pad=N_pad,unpad_result=unpad_result)*dx*N_x
    return E_u

def calc_ifft_pad(E_x,axis=0,N_pad=None,unpad_result=True):
    N_input = np.shape(E_x)[axis]
    if N_pad is not None:
        E_x = pad_array(E_x,N_pad,axis=axis)
    
    E_u = (fftshift(ifft(ifftshift(E_x,axes=axis),axis=axis),axes=axis))
    if unpad_result:
        E_u = unpad_array(E_u,N_input,axis=axis)
    return E_u

def pad_array(a,N_pad,axis=0,v=0):
    n_dim = len(np.shape(a))
    N_a = np.shape(a)[axis]
    n = N_pad-N_a
    p_list = [[0,0]]*n_dim
    p_list[axis]= [int(np.ceil(n/2)),int(np.floor(n/2))]
    if n_dim==1: 
        a_pad = np.pad(a,p_list[0],'constant',constant_values=v)
    else:
        a_pad = np.pad(a,p_list,'constant',constant_values=v)
    return a_pad

def calc_fraunhofer_axis(N,dx,N_pad=None,unpad_result=True):
    if N_pad is None:
        N_pad = N
    du = 1/((N_pad)*dx)
    u = (np.arange(N_pad)-N_pad/2)*du
    if unpad_result:
        u = unpad_array(u,N)
    return u