import numpy as np
from scipy.signal import butter, lfilter
from tqdm import tqdm


def get_bandranges(signal, bandranges, fs, filter_order):
    filt_signal = np.empty(tuple([len(bandranges)]) + signal.shape)
    for i,band in enumerate(bandranges):
        [b,a] = butter(filter_order,np.array(band)/(fs/2),'bandpass')
        filt_signal[i, :, :] = lfilter(b,a,signal,axis=0)
    return filt_signal


def get_sliding_covariance_trace_normalized(data, wlenght, wshift, substractWindowMean=True, dispProgress=True):
    nsamples, nchannels = data.shape
    
    wstart = np.arange(0, nsamples - wlenght + 1, wshift)
    wstop = wstart + wlenght
    
    nwins = len(wstart)
    
    C = np.empty((nwins, nchannels, nchannels))
    for wId in tqdm (range (nwins), bar_format='{l_bar}{bar:40}{r_bar}', disable=not dispProgress):        
        cstart = int(wstart[wId])
        cstop = int(wstop[wId])
        t_data = data[cstart:cstop, :]
        C[wId] = get_covariance_matrix_traceNorm(t_data, substractWindowMean)  
    return C


def get_covariance_matrix_traceNorm(data, substractWindowMean=True):
    if substractWindowMean:
        data -= np.mean(data, axis=0)
    t_cov = data.T @ data
    cov =  t_cov  / np.trace(t_cov)
    return cov


# # ---------------------- ONLINE ----------------------

def get_covariance_matrix_traceNorm_online(data):
    data -= np.mean(data, axis=(0,2), keepdims=True)
    cov = data.transpose((0,2,1)) @ data
    cov =  cov  / np.trace(cov, axis1=1, axis2=2).reshape(-1,1,1)
    cov = np.expand_dims(cov, axis=1)
    return cov