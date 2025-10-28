import numpy as np
import pandas as pd

import joblib
import os

from scipy.io import loadmat
import mne

from os.path import dirname
import tkinter as tk
from tkinter import filedialog as fd

from mne.io import read_raw_gdf
from mne import events_from_annotations


def saveto_npy(arr, pth):
    extension = 'npy'
    with open(pth, 'wb+') as fh:
        np.np_save(fh, arr, allow_pickle=False)
        fh.flush()
        os.fsync(fh.fileno())


def load_npy(pth):
    return np.np_load(pth)


def get_all_online_offline_files(path):
    directories = get_immediate_subdirectories(path)
    filepaths = []
    for dir in directories:
        filenames = get_filesNames_from_folder(path + '/' + dir)
        filenames = [path + '/' + dir + '/' + k for k in filenames if '.gdf' in k and ('online' in k or 'offline' in k)]
        if len(filenames)>0:
            filepaths += filenames
    signal, events_dataFrame = load_gdf_files(filepaths)
    return signal, events_dataFrame


def get_filesNames_from_folder(mypath, pattern=None):
    filenames = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    if pattern is not None:
        filenames = [f for f in filenames if pattern in f]
    return sorted(filenames)


def get_immediate_subdirectories(path_dir):
    subdirectories = [name for name in os.listdir(path_dir) if os.path.isdir(os.path.join(path_dir, name))]
    return sorted(subdirectories)


def get_files(path, ask_user=True, cutStartEnd=False):

    if ask_user:
        root = tk.Tk()
        filenames = ()
        while True:
            chosen_files = fd.askopenfilenames(initialdir=path)
            if len(chosen_files)==0:
                break
            filenames += chosen_files
        root.destroy() 
    else:
        filenames = path

    if filenames[0][-4:]=='.mat':
        signal, events_dataFrame, h, _ = load_mat_files(filenames, cutStartEnd=cutStartEnd)
    elif filenames[0][-4:]=='.gdf':
        signal, events_dataFrame, h = load_gdf_files(filenames)

    return signal, events_dataFrame, h, list(filenames)


def load_mat_files(filenames, cutStartEnd=False):
    signal = []
    eeg_dim_tot = 0
    d = dict()
    d['dur']=[]
    d['pos']=[]
    d['typ']=[]
    d['run']=[]
    d['day']=[]
    d['prt']=[]
    dates = []

    n_ses = 0
    last_day = ''
    for n,file in enumerate(filenames):
        print(' - Loading file: ' + file)
        data = loadmat(file_name=file)
        h = fix_mat(data['h'])

        if cutStartEnd:
            start = h['EVENT']['POS'][0]
            endS = h['EVENT']['POS'][-1] + h['EVENT']['DUR'][-1]
            data['s'] = data['s'][start:endS]
            h['EVENT']['POS'] -= start


        d['dur'].append(h['EVENT']['DUR'])
        d['typ'].append(h['EVENT']['TYP'])
        d['pos'].append(h['EVENT']['POS']+eeg_dim_tot-1)
        d['run'].append([n]*len(h['EVENT']['DUR']))
        if file.split('/')[-2] != last_day:     
            n_ses += 1
            t_day = file.split('/')[-2]
            dates.append(t_day[6:8]+'/'+t_day[4:6]+'/'+t_day[:4])
        d['day'].append([n_ses]*len(h['EVENT']['DUR']))
            
        if 'calibration' in file:
            d['prt'].append([0]*len(h['EVENT']['DUR']))
        elif 'evaluation' in file:
            d['prt'].append([1]*len(h['EVENT']['DUR']))
        elif 'control' in file:
            d['prt'].append([2]*len(h['EVENT']['DUR']))
        else:
            d['prt'].append([-1]*len(h['EVENT']['DUR']))
        last_day = file.split('/')[-2]
        signal.append(data['s'])
        eeg_dim_tot += data['s'].shape[0]


    signal = np.concatenate(signal, axis=0)
    d['dur'] = [ int(x) for x in np.concatenate(d['dur']) ]
    d['typ'] = [ int(x) for x in np.concatenate(d['typ']) ]
    d['pos'] = [ int(x) for x in np.concatenate(d['pos']) ]
    d['run'] = [ int(x) for x in np.concatenate(d['run']) ]
    d['day'] = [ int(x) for x in np.concatenate(d['day']) ]
    d['prt'] = [ int(x) for x in np.concatenate(d['prt']) ]
    #d['ses_vector'] = [ int(x) for x in np.concatenate(d['ses_vector']) ]
    
    events_dataFrame = pd.DataFrame(data=d)
    return signal, events_dataFrame, h, dates


def load_gdf_files(filenames):
    signal = []
    eeg_dim_tot = 0
    d = dict()
    d['pos']=[]
    d['typ']=[]
    d['run']=[]
    d['ses']=[]
    d['dur']=[]
    n_ses = 0
    last_day = ''
    for n,file in enumerate(filenames):
        print(' - Loading file: ' + file)
        eeg,h = read_gdf(file)
        d['typ'].append(h['EVENT']['TYP'])
        d['dur'].append(h['EVENT']['DUR'])
        d['pos'].append(h['EVENT']['POS']+eeg_dim_tot-1)
        d['run'].append([n]*len(h['EVENT']['TYP']))
        if file.split('/')[-2] == last_day or last_day=='':
            d['ses'].append([n_ses]*len(h['EVENT']['TYP']))
        else:
            n_ses += 1
            d['ses'].append([n_ses]*len(h['EVENT']['TYP']))
        last_day = file.split('/')[-2]
        signal.append(eeg[:,:-1])
        eeg_dim_tot += eeg.shape[0]

    signal = np.concatenate(signal, axis=0)
    d['typ'] = [ int(x) for x in np.concatenate(d['typ']) ]
    d['pos'] = [ int(x) for x in np.concatenate(d['pos']) ]
    d['run'] = [ int(x) for x in np.concatenate(d['run']) ]
    d['ses'] = [ int(x) for x in np.concatenate(d['ses']) ]
    d['dur'] = [ int(x) for x in np.concatenate(d['dur']) ]
    #d['ses_vector'] = [ int(x) for x in np.concatenate(d['ses_vector']) ]
    
    events_dataFrame = pd.DataFrame(data=d)
    return signal, events_dataFrame, h


def load(filename):
    if not filename.endswith('.joblib'):    filename += '.joblib'
    print(f"Loading model from {filename}...")
    return joblib.load(filename)

def save(filename, variable):
    if not filename.endswith('.joblib'):    filename += '.joblib'
    joblib.dump(variable, filename)


def read_gdf(spath,verbosity='error',raw_events=False):

    raw = read_raw_gdf(spath,verbose=verbosity)

    eeg = raw.get_data().T
    
    events,names = events_from_annotations(raw,verbose=verbosity)
    names = {v:int(k) for k,v in names.items()}
    events_pos = events[:,0]
    events_typ = [names[e] for  e in events[:,2]]

    events = {'POS':np.array(events_pos),'TYP':np.array(events_typ)}
    if not raw_events:
        events = get_events(events)

    header = {'SampleRate':raw.info['sfreq'],
              'EVENT': events,
              'ChannelNames':np.array(raw.info['ch_names']),
            }
    
    return eeg,header


def fix_mat(data):
    if data.dtype.names:
        new_data = dict()
        for name in data.dtype.names:
            new_data[name]=data[0][name][0]
        for k,v in new_data.items():
            if v.dtype.names:
                new_data[k] = fix_mat(v)
            else:
                new_data[k] = np.squeeze(v)
        return new_data
    else:
        return data
    


def get_events(events, OFFSET=0x8000):
    events = pd.DataFrame(events)
    true_events = events.loc[events['TYP']<OFFSET].copy() # Keep event openings only

    # Compute event ends based on the position of event closure
    true_events['END'] = np.nan

    for e in true_events['TYP'].unique():
        ev_idx = events['TYP']==(e + OFFSET)
        true_events.loc[true_events['TYP']==e,'END'] = np.where(events[ev_idx]['POS'],events[ev_idx]['POS'],0)

    true_events['END'] = true_events['END'].astype(int)
    true_events['DUR'] = true_events['END']-true_events['POS'] #Compute event duration

    # Keep only relevant columns
    true_events = true_events[['TYP','POS','DUR',]]

    return true_events.reset_index()

