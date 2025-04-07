import numpy as np
import pandas as pd
from py_utils.eeg_managment import proc_pos2win

from scipy.signal import butter, lfilter
from tqdm import tqdm


def get_trNorm_covariance_matrix(data, events, windowsLength, windowsShift, fc, substractWindowMean=True, dispProgress=True):
    cov_events = events.copy()
    if isinstance(events, pd.DataFrame):
        # [samples] --> [windows]
        cov_events['pos'] = proc_pos2win(cov_events['pos'], windowsShift*fc, 'backward', windowsLength*fc)
        cov_events['dur'] = [ int(x) for x in cov_events['dur']/(windowsShift*fc)+1 ]

    n_bandranges, nsamples, nchannels = data.shape

    nwindows = int((nsamples-windowsLength*fc)/(windowsShift*fc))+1

    Cov = np.empty((n_bandranges, nwindows, nchannels, nchannels))

    if dispProgress:
        print(' - Computing covariance matrices on the band ranges')
    for bId in range(n_bandranges):
        ccov = get_sliding_covariance_trace_normalized(data[bId],  windowsLength*fc, windowsShift*fc, substractWindowMean, dispProgress=dispProgress)    # covariances matrix
        Cov[bId] = ccov
    return  [Cov, cov_events]


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


def logbandpower(data, fs, slidingWindowLength=None):
    data_sqr = data ** 2
    if slidingWindowLength is not None:
        b = np.ones(slidingWindowLength * fs) / (slidingWindowLength * fs)
        data_sqr = lfilter(b, 1, data_sqr, axis=0)
    return np.log(data_sqr)


# # ---------------------- ONLINE ----------------------

def get_covariance_matrix_traceNorm_online(data):
    data -= np.mean(data, axis=(0,2), keepdims=True)
    cov = data.transpose((0,2,1)) @ data
    cov =  cov  / np.trace(cov, axis1=1, axis2=2).reshape(-1,1,1)
    cov = np.expand_dims(cov, axis=1)
    return cov