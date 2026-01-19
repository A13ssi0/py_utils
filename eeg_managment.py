import numpy as np
from sklearn.model_selection import train_test_split


def get_channelsMask(wantedChannels, actualChannels):
    wanted = [w.strip() for w in wantedChannels]
    actual = [a.strip() for a in actualChannels]
    return [item in wanted for item in actual]


def select_channels(signal, wantedChannels, actualChannels=[]):
    if signal.shape[1] == 32 and len(actualChannels)==0:
        actualChannels = np.array(['FP1', 'FP2', 'FZ', 'FC5', 'FC1', 'FC2', 'FC6',  'C3',  'CZ',  
            'C4', 'CP5', 'CP1', 'CP2', 'CP6',  'P3', 'Pz',  'P4',  'F1',  'F2', 'FC3', 
            'FCZ', 'FC4', 'C5',  'C1',  'C2',  'C6', 'CP3', 'CP4',  'P5',  'P1', 'P2',  'P6'])
        
    elif signal.shape[1] == 16 and len(actualChannels)==0:
        actualChannels =  np.array(['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3','C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz','CP2', 'CP4'])
    
    boolean_list = get_channelsMask(wantedChannels, actualChannels)
    return [signal[:,boolean_list], actualChannels[boolean_list]]


def proc_pos2win(POS, wshift, direction, wlength=-1):
    backward = False
    if direction=='forward':
        wlength = []
    elif direction=='backward':
        backward = True
        if wlength == -1:
            raise Exception("chk:arg', 'backward direction option requires to provide wlength")
    else:
        raise Exception('chk:arg', 'Direction not recognized: only forward and backward are allowed')
    wPOS = np.floor(POS/wshift) + 1
    if backward == True:
        wPOS = wPOS - np.floor(wlength/wshift)
    wPOS = [ int(x) for x in wPOS ]
    return wPOS


def get_EventsVector_onFeedback(events, lengthVector,  events_typ, column_name='typ', use_eog=True):    
    vector = np.full(lengthVector, np.nan)
    idx_events = np.array(events[np.isin(events['typ'], events_typ)].index)
    ev_type = events.loc[idx_events, column_name].values

    is_feedback = events.loc[idx_events+1, 'typ'] == 781
    is_feedback = is_feedback.values
    idx_events[is_feedback] += 1
    start = events.loc[idx_events, 'pos'].values
    duration = events.loc[idx_events, 'dur'].values

    for [t_start, t_duration, t_ev_type] in zip(start, duration, ev_type):
        vector[t_start:(t_start + t_duration)] = t_ev_type
    if use_eog:
        idx = events[events['typ'] == 1024].index
        start = events.loc[idx, 'pos'].values
        duration = events.loc[idx, 'dur'].values
        ev_type = events.loc[idx, column_name].values
        for [t_start, t_duration, t_ev_type] in zip(start, duration, ev_type):
            vector[t_start:(t_start + t_duration)] = np.nan
    return vector


def get_vector_onEvent(df_events, lengthVector, clmn_name):
    vector = np.full(lengthVector, np.nan)
    for [t_start, t_duration, t_ev_type] in zip(df_events['pos'], df_events['dur'], df_events[clmn_name]):
        vector[t_start:(t_start + t_duration)] = t_ev_type
    return vector


def get_indices_train_validation(events, classes, percVal = 0.2):
    idx_train = []
    idx_val = []
    for cl in classes:
        info_cues = events.loc[events['typ']==cl]
        train, val = train_test_split(info_cues, test_size=percVal)
        for idx_cue in train.index:
            feedback = events.loc[idx_cue+1]
            idx_train += list(( feedback['pos'] + range(feedback['dur'])) )
        for idx_cue in val.index:
            feedback = events.loc[idx_cue+1]
            idx_val += list(( feedback['pos'] + range(feedback['dur'])) )
    idx_train.sort()
    idx_val.sort()  
    return [idx_train, idx_val]


def apply_ROI_over_channels(data, channels, channelGroups, returnMean=True):
    if len(channelGroups)==0:
        return data
    if returnMean:
        newData = np.empty((data.shape[0], len(channelGroups)))
    else:
        newData = np.empty(len(channelGroups), dtype=object)
    for nchs,chs in enumerate(channelGroups):
        _,idx,_ = np.intersect1d(channels, chs, return_indices=True)
        if returnMean:
            newData[:,nchs] = np.mean(data[:,idx], axis=1)
        else:
            newData[nchs] = data[:,idx]
    return newData


def fromEvents2vector(length_vector, events_name, events):
    vector = np.full(length_vector, np.nan)
    for idx in range(len(events[events_name])):
        start = events.loc[idx, 'pos']
        duration = events.loc[idx, 'dur']
        vector[start:(start + duration)] = events.loc[idx, events_name]
    return vector