import numpy as np
import pandas as pd
from py_utils.eeg_managment import proc_pos2win
from matplotlib import mlab
from scipy import signal
import warnings
from scipy.signal import butter, lfilter, lfilter_zi, tf2zpk
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
        # Check stability
        z, p, k = tf2zpk(b, a)  # Get zeros, poles, and gain
        stable = np.all(np.abs(p) < 1)
        if not stable:
            warnings.warn(f'[get_bandranges] Warning: The filter for band {band} is unstable!', category=UserWarning)
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
    if (data_sqr == 0).any():
        print("!!! WARNING: zero values found when computing logbandpower, adding 1e-12 to avoid log(0) !!!")
        data_sqr += 1e-12
    return np.log(data_sqr)



def proc_spectrogram(data, wlength, wshift, pshift, samplerate, mlength=None):
    """
    [features, f] = proc_spectrogram(data, wlength, wshift, pshift, samplerate [, mlength])
    
    The function computes the spectrogram on the real data.
    
    Input arguments:
        - data              Data matrix [samples x channels]
        - wlength           Window's lenght to be used to segment data and
                           compute the spectrogram                             [in seconds]
        - wshift            Shift of the external window (e.g., frame size)     [in seconds]
        - pshift            Shift of the internal psd windows                   [in seconds]
        - samplerate        Samplerate of the data
        - [mlength]         Optional length of the external windows to compute
                           the moving average.                                 [in seconds] 
                           By default the length of the moving average window
                           is set to 1 second. To not compute the moving
                           average, empty argument can be provided.
    
    Output arguments:
        - features          Output of the spectrogram in the format: 
                           [windows x frequencies x channels]. Number of
                           windows (segments) is computed according to the
                           following formula: 
                           nsegments = fix((NX-NOVERLAP)/(length(WINDOW)-NOVERLAP))
                           where NX is the total number of samples, NOVERLAP
                           the number of overlapping samples for each segment
                           and length(WINDOW) the number of samples in each
                           segment. 
                           Number of frequencies is computed according to the
                           NFFT. nfrequencies is equal to (NFFT/2+1) if NFFT 
                           is even, and (NFFT+1)/2 if NFFT is odd. NFFT is the
                           maximum between 256 and the next power of 2 greater
                           than the length(WINDOW).
        - f                 Vectore with the computed frequencies
    
    """
    
    # Data informations
    nsamples = data.shape[0]
    nchannels = data.shape[1]

    # Useful params for PSD extraction with the fast algorithm
    psdshift = pshift * samplerate
    winshift = wshift * samplerate

    if (psdshift % winshift != 0) and (winshift % psdshift != 0):
        warnings.warn('[proc_spectrogram] The fast PSD method cannot be applied with the current settings!', 
                     category=UserWarning)
        raise ValueError('[proc_spectrogram] The internal welch window shift must be a multiple of the overall feature window shift (or vice versa)!')

    # Create arguments for spectrogram
    spec_win = int(wlength * samplerate)
    
    # Careful here: The overlapping depends on whether the winshift or the
    # psdshift is smaller. Some calculated internal windows will be redundant,
    # but the speed is much faster anyway

    if psdshift <= winshift:
        spec_ovl = spec_win - int(psdshift)
    else:
        spec_ovl = spec_win - int(winshift)

    # Calculate all the internal PSD windows
    nsegments = int((nsamples - spec_ovl) / (spec_win - spec_ovl))  # From spectrogram's help page
    nfft = max(256, int(2**(np.ceil(np.log2(spec_win)))))  # nextpow2 equivalent
    if nfft % 2 == 0:
        nfreqs = (nfft // 2) + 1
    else:
        nfreqs = (nfft + 1) // 2
    
    psd = np.zeros((nfreqs, nsegments, nchannels))
    window = np.hamming(samplerate* wlength)  # Hamming window for the spectrogram
    #NOverlap = wlength * samplerate  # Overlap for the spectrogram

    for chId in range(nchannels):
        #f, t, Sxx = signal.spectrogram(data[:, chId], fs=samplerate, window='hamming', 
        #                              nperseg=spec_win, noverlap=spec_ovl, nfft=nfft)
        #[~,f,~,psd(:,:,chId)] = spectrogram(data(:,chId), spec_win, spec_ovl, [], samplerate)
       [psd[:, :, chId], f, t] = mlab.specgram(data[:, chId], NFFT = nfft, Fs = samplerate, window = window, noverlap = spec_ovl)
        #psd[:, :, chId] = Sxx
    
    if mlength is not None:
        # Setup moving average filter parameters
        mavg_a = 1
        if winshift >= psdshift:
            # Case where internal windows are shifted according to psdshift
            mavgsize = int(((mlength * samplerate) / psdshift) - 1)
            mavg_b = (1 / mavgsize) * np.ones(mavgsize)
            mavg_step = int(winshift / psdshift)
        else:
            # Case where internal windows are shifted according to winshift
            mavgsize = int(((mlength * samplerate) / winshift) - (psdshift / winshift))
            mavg_b = np.zeros(mavgsize)
            step_size = int(psdshift / winshift)
            mavg_b[0:mavgsize-1:step_size] = 1
            mavg_b = mavg_b / np.sum(mavg_b)
            mavg_step = 1
        
        # Find last non-zero element (equivalent to find(mavg_b~=0, 1, 'last'))
        startindex = np.where(mavg_b != 0)[0][-1]

        # Apply filter along axis 1 (equivalent to filter(mavg_b,mavg_a,psd,[],2))
        features = signal.lfilter(mavg_b, mavg_a, psd, axis=1)
        # Permute dimensions: [2 1 3] -> transpose from (nfreqs, nsegments, nchannels) to (nsegments, nfreqs, nchannels)
        features = np.transpose(features, (1, 0, 2))

        # Get rid of initial filter byproducts
        features = features[startindex:, :, :]

        # In case of psdshift, there will be redundant windows. Remove them
        if mavg_step > 1:
            features = features[::mavg_step, :, :]
    else:
        features = psd
        # Permute dimensions: [2 1 3] -> transpose from (nfreqs, nsegments, nchannels) to (nsegments, nfreqs, nchannels)
        features = np.transpose(features, (1, 0, 2))
    
    return features, f


def compute_fisher_score(psd, freqs, classes, runVector, runs_labels, isCFeedbackVector=None, validRuns=None, SelFreqs=None):

    if SelFreqs is None:    SelFreqs = freqs
    idfreqs = np.nonzero(np.isin(freqs, SelFreqs))[0]
    freqs = freqs[idfreqs]

    if isCFeedbackVector is None:   isCFeedbackVector = np.ones(runVector.shape, dtype=bool)
    if validRuns is None:           validRuns = np.unique(runVector[isCFeedbackVector]).astype(int)
    n_runs = validRuns.shape[0]


    u = psd[:, idfreqs, :] 
    NumWins, NumFreqs, NumChans = u.shape

    # --- Computing Fisher score (for each run) ---
    # print("[proc] + Computing fisher score")

    FisherScore = np.full((NumFreqs, NumChans, n_runs), np.nan)
    cva = FisherScore.copy()

    for count_run, nR in tqdm (enumerate(validRuns), total=n_runs, bar_format='{l_bar}{bar:40}{r_bar}'):
        # print('count: ' + str(count_run), '  nR: ' + str(nR), end="  ")
        chosen_idx = (runVector==nR) & (isCFeedbackVector)
        lbl = runs_labels[nR-1]

        counter = 0
        while len(lbl) == 1 and counter < 10:
            lbl = lbl[0]
            counter += 1

        data = u[chosen_idx]
        cmu = np.full((NumFreqs, NumChans, 2), np.nan)
        csigma = np.full((NumFreqs, NumChans, 2), np.nan)

        flag = np.full((len(classes)), False)
        for cId, clss in enumerate(classes):
            if np.sum(lbl==clss) > 0:
                flag[cId] = True
                # print(str(clss), end="  ")
                cmu[:, :, cId] = np.nanmean(data[lbl==clss], axis=0)
                csigma[:, :, cId] = np.nanstd(data[lbl==clss], axis=0)

        if not all(flag):
            raise ValueError(f'Missing classes in  run {nR}: classes not present {classes[flag]}')
        
        FisherScore[:, :, count_run] = np.abs(cmu[:, :, 1] - cmu[:, :, 0]) / np.sqrt(
            csigma[:, :, 0]**2 + csigma[:, :, 1]**2
        )
                        
        u_flat = data.reshape(data.shape[0], -1, order='F')  # Flatten freq*chan
        cva_result, _, _, _, _ = cva_tun_opt(u_flat, lbl)
        cva[:, :, count_run] = cva_result.reshape(NumFreqs, NumChans)

    return FisherScore, cva






# def compute_fisher_score(psd, events, freqs, runVector, selectedRuns=None, classes=np.array([771, 773]), bool_dayVector=None, bool_protocolVector=None, SelFreqs=None):

#     if SelFreqs is None:    SelFreqs = freqs
#     if bool_dayVector is None:    bool_dayVector = np.ones(runVector.shape, dtype=bool)
#     if bool_protocolVector is None:    bool_protocolVector = np.ones(runVector.shape, dtype=bool)

#     if selectedRuns is not None:
#         events = events[ np.isin(events.run, selectedRuns) ].reset_index(drop=True)

#     nwindows, nfreqs, nchannels = psd.shape

#     pos = events.pos
#     dur = events.dur
#     typ = events.typ

#     # --- Creating vector labels ---

#     CFeedbackPOS = pos[typ == 781].reset_index(drop=True)
#     CFeedbackDUR = dur[typ == 781].reset_index(drop=True)

#     CueMask = (typ == 771) | (typ == 773) | (typ == 783)
#     CuePOS = pos[CueMask].reset_index(drop=True)
#     # CueDUR = dur[CueMask]
#     CueTYP = typ[CueMask].reset_index(drop=True)

#     # FixPOS = pos[typ == 786]
#     # FixDUR = dur[typ == 786]
#     # FixTYP = typ[typ == 786]

#     NumTrials = len(CFeedbackPOS)

#     # --- Consider interesting period from Cue appearance to end of continuous feedback ---

#     Ck = np.zeros(nwindows, dtype=float)
#     Tk = np.zeros(nwindows, dtype=float)
#     TrialStart = np.full(NumTrials, np.nan)
#     TrialStop = np.full(NumTrials, np.nan)

#     for trId in range(NumTrials):
#         cstart = int(CuePOS[trId])
#         cstop = int(CFeedbackPOS[trId] + CFeedbackDUR[trId] - 1)
#         Ck[cstart:cstop+1] = CueTYP[trId]
#         Tk[cstart:cstop+1] = trId
        
#         TrialStart[trId] = cstart
#         TrialStop[trId] = cstop

#     # --- Apply log to data (already done it) ---

#     # freqs, SelFreqs assumed as numpy arrays
#     idfreqs = np.nonzero(np.isin(freqs, SelFreqs))[0]
#     freqs = freqs[idfreqs]
#     u = psd[:, idfreqs, :]  # shape: (NumWins, NumFreqs, NumChans)

#     NumWins, NumFreqs, NumChans = u.shape

#     # --- Select wanted day ---
#     if selectedRuns is None:
#         Runs = np.unique(runVector[bool_dayVector & bool_protocolVector])
#     else:
#         Runs = selectedRuns

#     NumRuns = len(Runs)
#     print(f"Found {NumRuns} runs")

#     # --- Computing Fisher score (for each run) ---
#     print("[proc] + Computing fisher score")
#     NumClasses = len(classes)

#     FisherScore = np.full((NumFreqs, NumChans, NumRuns), np.nan)
#     cva = np.full_like(FisherScore, np.nan)
#     skip_run = 0

#     for rId, run in enumerate(Runs):
#         rindex = (runVector == run)

#         cmu = np.full((NumFreqs, NumChans, 2), np.nan)
#         csigma = np.full((NumFreqs, NumChans, 2), np.nan)

#         for cId, cls in enumerate(classes):
#             cindex = rindex & (Ck == cls)
#             if not np.any(cindex):
#                 print(f"Warning: No data for class {cls} in run {run}")
#                 skip_run += 1
#                 continue

#             cmu[:, :, cId] = np.nanmean(u[cindex, :, :], axis=0)
#             csigma[:, :, cId] = np.nanstd(u[cindex, :, :], axis=0)

#         if skip_run==NumClasses:
#             skip_run = 0
#             continue

#         FisherScore[:, :, rId] = np.abs(cmu[:, :, 1] - cmu[:, :, 0]) / np.sqrt(
#             csigma[:, :, 0]**2 + csigma[:, :, 1]**2
#         )

#         cindex = rindex & np.isin(Ck, classes)
#         u_sel = u[cindex, :, :]                           
#         u_flat = u_sel.reshape(u_sel.shape[0], -1, order='F')  # Flatten freq*chan
#         cva_result, _, _, _, _ = cva_tun_opt(u_flat, Ck[cindex])
#         # cva_result = cva_tun_opt(u[cindex, :, :], Ck[cindex])  # You must define this
#         cva[:, :, rId] = cva_result.reshape(NumFreqs, NumChans)

#     return FisherScore, cva



def cva_tun_opt(pat, label):
    """
    Canonical Variate Analysis (CVA) transformation.

    Inputs:
        pat   : ndarray (n_samples x n_features)
        label : ndarray (n_samples,)
    
    Outputs:
        com   : Discriminability Power (%) of each feature
        pwgr  : Within-group correlation matrix (features x components)
        v     : Eigenvectors (features x components)
        vp    : Eigenvalues (vector)
        disc  : Transformed data (canonical variates)
    """

    # Ensure numpy arrays
    pat = np.asarray(pat)
    label = np.asarray(label)

    # --- Relabel classes to 1..n_class
    labels_old = np.unique(label)
    label_new = np.zeros_like(label, dtype=int)
    for j, lab in enumerate(labels_old):
        label_new[label == lab] = j + 1
    label = label_new

    n_feature = pat.shape[1]
    n_class = int(label.max())

    # --- Within-class covariance matrix
    cov_w = np.zeros((n_feature, n_feature, n_class))
    for k in range(1, n_class + 1):
        class_data = pat[label == k, :]
        if class_data.shape[0] > 1:
            cov_w[:, :, k - 1] = (class_data.shape[0] - 1) * np.cov(class_data, rowvar=False)
    cov_total = np.sum(cov_w, axis=2)
    mean_total = np.mean(pat, axis=0)

    # --- Between-class covariance matrix
    cov_b = np.zeros((n_feature, n_feature, n_class))
    for k in range(1, n_class + 1):
        class_data = pat[label == k, :]
        cent_g = np.mean(class_data, axis=0) - mean_total
        cov_b[:, :, k - 1] = class_data.shape[0] * np.outer(cent_g, cent_g)
    b_total = np.sum(cov_b, axis=2)

    # --- Solve generalized eigenproblem
    c_inv = np.linalg.pinv(cov_total)
    matrix = c_inv @ b_total
    u, s, vt = np.linalg.svd(matrix)
    v = u
    vp = np.diag(s) if s.ndim == 2 else s

    disc = pat @ v[:, :n_class - 1]

    # --- Within-group correlation matrix
    wg_cov = np.zeros((n_class - 1 + n_feature, n_class - 1 + n_feature, n_class))
    for k in range(1, n_class + 1):
        class_data = np.hstack([disc[label == k, :], pat[label == k, :]])
        if class_data.shape[0] > 1:
            wg_cov[:, :, k - 1] = (class_data.shape[0] - 1) * np.cov(class_data, rowvar=False)
    pwg_cov = np.sum(wg_cov, axis=2)
    pwgr = pwg_cov[n_class - 1:, :n_class - 1].copy()

    # Normalize to get correlation coefficients
    for i in range(pwgr.shape[0]):
        for u_idx in range(pwgr.shape[1]):
            denom = np.sqrt(pwg_cov[i + n_class - 1, i + n_class - 1] * pwg_cov[u_idx, u_idx])
            pwgr[i, u_idx] = pwgr[i, u_idx] / denom if denom != 0 else 0

    # --- Discriminability Power (%)
    vp = vp[:n_class - 1]
    vp_norm = vp / np.sum(vp)
    com = 100.0 * ((pwgr ** 2) @ vp_norm) / np.sum((pwgr ** 2) @ vp_norm)

    return com, pwgr, v, vp, disc



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
                    for ch in range(data_chunk.shape[1])]).T

        y, self.zi = lfilter(self.b, self.a, data_chunk, axis=0, zi=self.zi)
        return y
    

class RealTimeLogBandPower:
    def __init__(self, fs, sliding_window_length=None):
        self.fs = fs
        self.zi = None 
        if sliding_window_length is not None:
            n_points = sliding_window_length * fs
            self.b = np.ones(n_points) / n_points
            self.a = [1.0]
        else:
            self.b, self.a = None, None

    def process(self, data_chunk):  
        data_sqr = data_chunk ** 2

        if self.zi is None and self.b is not None:
            if data_chunk.ndim == 1:
                self.zi = lfilter_zi(self.b, self.a) * data_chunk[0]
            else:
                self.zi = np.array([
                    lfilter_zi(self.b, self.a) * data_chunk[0, ch]
                    for ch in range(data_chunk.shape[1])]).T
                
        if self.b is not None:
            data_sqr, self.zi = lfilter(self.b, self.a, data_sqr, axis=0, zi=self.zi)
        if (data_sqr == 0).any():
            print("!!! WARNING: zero values found when computing logbandpower, adding 1e-12 to avoid log(0) !!!")
            data_sqr += 1e-12
        return np.log(data_sqr)


    
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


