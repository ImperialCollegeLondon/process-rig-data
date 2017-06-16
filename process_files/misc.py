# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

GECKO_DT_FMT = '%d%m%Y_%H%M%S' 

def read_rig_csv_db(csv_file):
    if csv_file.endswith('.csv'):
        db = pd.read_csv(csv_file)
        db.columns = [x.strip() for x in db.columns]
        if db.columns.size == 1:
            db = pd.read_table(csv_file)
    else:
        db = pd.read_excel(csv_file)
    
    
    db.dropna(inplace=True, how='all')
    db.columns = [x.lower() for x in db.columns]
    
    assert all(x in db for x in ['set_n', 'rig_pos', 'camera_n'])
    
    db[['set_n', 'rig_pos', 'camera_n']] = db[['set_n', 'rig_pos', 'camera_n']].astype(np.int)
    
    db_ind = {(row['set_n'],row['rig_pos'],row['camera_n']) : irow 
             for irow, row in db.iterrows()}
          
    if db.shape[0] == len(db_ind):
        return db, db_ind
    else:
        raise ValueError('There must be only one combination set_n, rig_pos, camera_n per sample in the .csv file')

def gecko_fnames_to_table(filenames):
    def _gecko_movie_parts(x):
        '''
        Read movie parts as expected from a Gecko file.
        '''
        dir_name, bb = os.path.split(x)
        base_name, ext = os.path.splitext(bb)
        
        base_name = base_name.lower()
        base_name = base_name.replace('set_', 'set')
        
        parts = base_name.split('_')
        
        if parts[-1].isalpha():
            postfix = '_' + parts[-1]
            parts = parts[:-1]
        else:
            postfix = ''
        
        try:
            video_timestamp = pd.to_datetime(parts[-2] + '_' + parts[-1], 
                                   format=GECKO_DT_FMT)
            parts = parts[:-2]
        
        except ValueError:
            video_timestamp = pd.NaT
            
     
        def _part_n(start_str, parts_d):
            if parts[-1].startswith(start_str):
                part_n = int(parts_d[-1].replace(start_str, ''))
                parts_d = parts_d[:-1]
            else:
                part_n = -1
            return part_n, parts_d
        
            
        channel, parts = _part_n('ch', parts)
        stage_pos, parts = _part_n('pos', parts)
        set_n, parts = _part_n('set', parts)
        
        
        prefix = '_'.join(parts)
        
        return dir_name, base_name, ext, prefix, channel, stage_pos, set_n, video_timestamp, postfix
        
    fparts = [_gecko_movie_parts(x) for x in filenames]
    fparts_tab = pd.DataFrame(fparts, columns = ('directory', 'base_name', 'ext', 'prefix', 'channel', 'stage_pos', 'set_n', 'video_timestamp', 'postfix'))
    return fparts_tab



def _get_set_delta_t(experiments):
    #correct each video timestamp by the first experiment in the set (this is usefull if several videos were taken of the same sample)
    def _delta_timestamp(video_timestamp):
        delT = video_timestamp - video_timestamp.min()
        delT /= np.timedelta64(1, 'm')
        return delT

    set_dT = pd.Series()
    exp_dT = pd.Series()
    groupby_exp = experiments.groupby('exp_name')
    for exp_name, exp_rows in groupby_exp:
        delT = _delta_timestamp(exp_rows['video_timestamp'])
        exp_dT = exp_dT.append(delT)
        for set_n, set_rows in exp_rows.groupby('set_n'):
            delT = _delta_timestamp(set_rows['video_timestamp'])
            set_dT = set_dT.append(delT)
        
    experiments['set_delta_time'] = set_dT
    experiments['exp_delta_time'] = exp_dT
    return experiments

def get_rig_experiments_df(features_files, csv_files):
    '''
    Get experiments data from the files located in the main directory and from
    the experiments database
    '''
    
    def _read_db(x):
        db, _ = read_rig_csv_db(x)
        db.columns = [x.strip() for x in db.columns]
        exp_name = os.path.basename(x).replace('.csv', '').replace('.xlsx', '')
        db['exp_name'] = exp_name
        return db
    
    db_csv = pd.concat([_read_db(x) for x in csv_files], ignore_index=True)
    db_csv.rename(columns={'set_n': 'set_n', 'camera_n': 'channel' , 'rig_pos':'stage_pos'}, inplace=True)
    
    
    fparts_tab = gecko_fnames_to_table([x.replace('_features.hdf5', '.hdf5') for x in features_files ])
    fparts_tab['exp_name'] = fparts_tab['directory'].apply(lambda x : x.split(os.sep)[-1])
    
    experiments = pd.merge(fparts_tab, db_csv, on=['exp_name', 'set_n', 'channel', 'stage_pos'])
    experiments = _get_set_delta_t(experiments)
    return experiments

