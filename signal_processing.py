import numpy as np
import pandas as pd
from py_utils.eeg_managment import proc_pos2win
from scipy.signal import butter, lfilter, lfilter_zi
from tqdm import tqdm
from pyriemann.utils.covariance import covariances

def get_covariance_matrix_normalized(data, events, windowsLength, windowsShift, fs, normalizationMethod='lwf', substractWindowMean=True, dispProgress=True):
    cov_events = events.copy()
    if isinstance(events, pd.DataFrame):
        # [samples] --> [windows]
        cov_events['pos'] = proc_pos2win(cov_events['pos'], windowsShift*fs, 'backward', windowsLength*fs)
        cov_events['dur'] = [ int(x) for x in cov_events['dur']/(windowsShift*fs)+1 ]

    n_bandranges, nsamples, nchannels = data.shape

    nwindows = int((nsamples-windowsLength*fs)/(windowsShift*fs))+1

    Cov = np.empty((n_bandranges, nwindows, nchannels, nchannels))

    if dispProgress:
        print(' - Computing covariance matrices on the band ranges')
    for bId in range(n_bandranges):
        ccov = get_sliding_covariance_normalized(data[bId],  windowsLength*fs, windowsShift*fs, substractWindowMean=substractWindowMean, dispProgress=dispProgress, normalizationMethod=normalizationMethod)    # covariances matrix
        Cov[bId] = ccov
    return  [Cov, cov_events]



def get_trNorm_covariance_matrix(data, events, windowsLength, windowsShift, fs, substractWindowMean=True, dispProgress=True):
    cov_events = events.copy()
    if isinstance(events, pd.DataFrame):
        # [samples] --> [windows]
        cov_events['pos'] = proc_pos2win(cov_events['pos'], windowsShift*fs, 'backward', windowsLength*fs)
        cov_events['dur'] = [ int(x) for x in cov_events['dur']/(windowsShift*fs)+1 ]

    n_bandranges, nsamples, nchannels = data.shape

    nwindows = int((nsamples-windowsLength*fs)/(windowsShift*fs))+1

    Cov = np.empty((n_bandranges, nwindows, nchannels, nchannels))

    if dispProgress:
        print(' - Computing covariance matrices on the band ranges')
    for bId in range(n_bandranges):
        ccov = get_sliding_covariance_normalized(data[bId],  windowsLength*fs, windowsShift*fs, substractWindowMean=substractWindowMean, dispProgress=dispProgress, normalizationMethod='trace')    # covariances matrix
        Cov[bId] = ccov
    return  [Cov, cov_events]


def get_bandranges(signal, bandranges, fs, filter_order, filtType):
    if len(bandranges) == 0:    return signal

    if len(signal.shape) == 2:  filt_signal = np.empty(tuple([len(bandranges)]) + signal.shape)
    elif len(signal.shape) == 3:  filt_signal = np.empty(signal.shape)

    for i,band in enumerate(bandranges):
        [b,a] = butter(filter_order,np.array(band)/(fs/2), filtType)
        filt_signal[i, :, :] = lfilter(b,a,signal,axis=0)
    return filt_signal


def get_sliding_covariance_normalized(data, wlenght, wshift, normalizationMethod='trace', substractWindowMean=True, dispProgress=True):
    nsamples, nchannels = data.shape
    
    wstart = np.arange(0, nsamples - wlenght + 1, wshift)
    wstop = wstart + wlenght
    
    nwins = len(wstart)
    
    c = np.empty((nwins, nchannels, nchannels))
    for wId in tqdm (range (nwins), bar_format='{l_bar}{bar:40}{r_bar}', disable=not dispProgress):        
        cstart = int(wstart[wId])
        cstop = int(wstop[wId])
        t_data = data[cstart:cstop, :]
        if normalizationMethod=='trace' : c[wId] = get_covariance_matrix_traceNorm(t_data, substractWindowMean)  
        elif normalizationMethod=='lwf' : c[wId] = covariances(np.expand_dims(t_data.T, axis=0), estimator='lwf')[0]
    return c




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
class RealTimeButterFilter:
    def __init__(self, order, cutoff, fs, type):
        self.order = order
        self.cutoff = cutoff
        self.fs = fs
        self.b, self.a = butter(order, 2 * cutoff / fs, btype=type)
        self.zi = None  # will be initialized on first call

    def filter(self, data_chunk):
        if self.zi is None:
            # Initialize zi for each channel if data is 2D
            if data_chunk.ndim == 1:
                self.zi = lfilter_zi(self.b, self.a) * data_chunk[0]
            else:
                self.zi = np.array([
                    lfilter_zi(self.b, self.a) * data_chunk[0, ch]
                    for ch in range(data_chunk.shape[1])
                ]).T

        y, self.zi = lfilter(self.b, self.a, data_chunk, axis=0, zi=self.zi)
        return y
    
def get_covariance_matrix_traceNorm_online(data):
    if data.ndim == 2:  data = np.expand_dims(data, axis=0)
    data -= np.mean(data, axis=(0, 1), keepdims=True)
    cov = data.transpose((0, 2, 1)) @ data
    # print('cov,',cov.shape)
    # cov = covariances(data.transpose((0, 2, 1)), estimator='lwf')
    cov =  cov  / np.trace(cov, axis1=1, axis2=2).reshape(-1,1,1)
    return np.expand_dims(cov, axis=1)

def get_covariance_matrix_lwfNorm_online(data):
    if data.ndim == 2:  data = np.expand_dims(data, axis=0)
    data -= np.mean(data, axis=(0, 1), keepdims=True)
    cov = covariances(data.transpose((0, 2, 1)), estimator='lwf')
    return np.expand_dims(cov, axis=1)


