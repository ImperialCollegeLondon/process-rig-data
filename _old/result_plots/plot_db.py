#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 08:19:55 2016

@author: ajaver
"""

import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from sqlalchemy import create_engine
import numpy as np

def _delta_timestamp(video_timestamp):
    delT = video_timestamp - video_timestamp.min()
    delT /= np.timedelta64(1, 'm')
    return delT

def _get_set_delta_t(experiments):
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

def _get_n_worms_estimate(experiments, feats):
    n_worms_dat = []
    for video_id, video_feats in feats.groupby('video_id'):
        first_frame = video_feats['first_frame'].astype(np.int)
        last_frame = first_frame + video_feats['n_frames'].astype(np.int)
        
        tot_frames = last_frame.max()
        n_worms = np.zeros(tot_frames, np.int)
        
        for ini, fin in zip(first_frame, last_frame):
            n_worms[ini:fin] += 1
        
        n_99 = np.percentile(n_worms, [99])[0]
        n_worms_dat.append((video_id, n_99, tot_frames))
    
    video_ids, n_worms, tot_frames = zip(*n_worms_dat)
        
    n_worms_estimate = pd.Series(n_worms, index = video_ids)
    tot_frames = pd.Series(n_worms, index = video_ids)
    
    experiments['n_worms_estimate'] = n_worms_estimate
    experiments['tot_frames'] = tot_frames
    return experiments
    

def _get_delta_start(experiments):
    def _get_deltaT(video_timestamp):
        deltaT = video_timestamp - video_timestamp.min()
        deltaT /= np.timedelta64(1, 'm')
        return deltaT
    
    dT_exp = pd.Series()
    dT_set = pd.Series()
    groupby_exp = experiments.groupby('exp_name')
    for exp_name, exp_rows in groupby_exp:
        deltaT = _get_deltaT(exp_rows['video_timestamp'])
        dT_exp = dT_exp.append(deltaT)
        for set_n, set_rows in exp_rows.groupby('set_n'):
            deltaT = _get_deltaT(set_rows['video_timestamp'])
            dT_set = dT_set.append(deltaT)
    
    experiments['delta_start_exp'] = dT_exp
    experiments['delta_start_set'] = dT_set  
    
    return experiments
#%%
def _get_observed_n_worms(feats, experiments):
    group_by_video =   feats.groupby('video_id')
    
    n_worms_dat = []
    for video_id, video_dat in group_by_video:
        ini_frame = video_dat['first_frame'].astype(np.int)
        fin_frame = video_dat['n_frames'].astype(np.int)
        
        tot_frames = fin_frame.max()
        n_worms_f = np.zeros(tot_frames, dtype = np.int)
        
        for ini, fin in zip(ini_frame, fin_frame):
            n_worms_f[ini:fin] += 1
        
        n_worms = np.percentile(n_worms_f, 99.5)
        n_worms_dat.append((video_id, n_worms, tot_frames))
    
    dd = list(zip(*n_worms_dat))
    observed_n_worms = pd.DataFrame(np.array([dd[1], dd[2]]).T, dd[0], columns=['observed_n_worms', 'tot_frames'])
    experiments = experiments.join(observed_n_worms)
    return experiments
#%%

def _add_sample_id(experiments):
    
    col_keys = ['exp_name', 'set_n', 'stage_pos', 'channel']
    
    if any(not x in experiments for x in col_keys):
        return experiments
    
    dd = experiments[col_keys]
    dd = dd.drop_duplicates()
    #dd.sort_values(col_keys, inplace=True)
    dd.index = np.arange(len(dd))
    
    sample_key = {tuple(row.values):sample_id for sample_id, row in dd.iterrows()};
    
    samp_tup = ((ind, sample_key[tuple(row.values)]) for ind, row in experiments[col_keys].iterrows())
    index, sample_id = zip(*samp_tup)
    
    experiments['sample_id'] = pd.Series(sample_id, index)
    return experiments 

    
def _filter_feats(con, feats, filt_path_range = 10, filt_frac_good = 0.75, tab_name = 'means'):
    #
    feats_ind = pd.read_sql_query('SELECT worm_index, n_frames, n_valid_skel, path_range, video_id FROM %s' % tab_name, con)
    feats_ind['frac_good'] = feats_ind['n_valid_skel']/feats_ind['n_frames']
    good = (feats_ind['path_range'] > filt_path_range) & (feats_ind['frac_good']>filt_frac_good)
    good_ind = feats_ind.loc[good, ['worm_index', 'video_id']]
    good_ind = [tuple(x) for x in good_ind.values.astype(np.int)]
    #all the tuples are uniqu
    assert len(good_ind) == len(set(good_ind))

    good_rows = [(row['worm_index'], row['video_id']) in good_ind for ii, row in feats.iterrows()]
    
    return good_rows
                 
def _read_feats(con, tab_name = 'split'):
    experiments = pd.read_sql_query('SELECT * FROM experiments', con)
    _add_sample_id(experiments)
    
    if 'video_timestamp' in experiments:
        experiments['video_timestamp'] = pd.to_datetime(experiments['video_timestamp'])
        experiments['date'] = experiments['video_timestamp'].dt.date
        experiments = _get_set_delta_t(experiments)
     
    
    feats = pd.read_sql_query('SELECT * FROM %s' % tab_name, con)
    #experiments = _get_observed_n_worms(feats, experiments)
    
    feats = feats.merge(experiments, on='video_id')
    feats['start_time'] = feats['first_frame']/25/60
    
    experiments = _get_n_worms_estimate(experiments, feats)
    
    if 'worm_index' in feats:
        feats['worm_index'] = feats['worm_index'].astype(np.int)
    
    return feats, experiments
                 
def get_feats_db(database_name, filt_path_range = 10, filt_frac_good = 0.75, 
                 tab_name = 'means_split'):
    con = create_engine('sqlite:///' + database_name)
    
    feats, experiments = _read_feats(con, tab_name)
    
    #let's use the full features to do the filtering
    tab_name_c = tab_name.replace('_split', '')
    good_rows = _filter_feats(con, feats, filt_path_range, filt_frac_good, tab_name = tab_name_c)
    
    return feats.loc[good_rows, :], experiments
    
#%%
def plot_boxes(feats, feat_str, main_div, sub_div = None):
    #violinplot
    
    if sub_div is None:
        ff = sns.boxplot(feats[main_div], feats[feat_str])
    else:
        ff = sns.boxplot(feats[main_div], feats[feat_str], hue=feats[sub_div])
    #ff = sns.violinplot(feats[main_div], feats[feat_str], hue=feats[sub_div])
    ff.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return ff

def plot_boxes_group(feats, feat_str, main_div, sub_div, group_div, feat_ylim=[]):
    plt.figure()
    
    groups = feats.groupby(group_div)
    tot_g = len(groups)
    for ii, (picker, feats_p) in enumerate(groups):
        feats_p = feats_p.sort_values(by=[main_div,  sub_div], ascending=False)
        
        plt.subplot(tot_g,1, ii + 1)
        ff = plot_boxes(feats_p, feat_str, main_div, sub_div)
        
        plt.title(picker)
        if ii < len(groups)-1:
            ff.get_xaxis().set_ticklabels([])
            ff.set_xlabel('')
        
            
        if len(feat_ylim) == 2:
            ff.set_ylim(feat_ylim)

            
class plot_db(object):
    def __init__(self, db_path, filt_path_range, filt_frac_good, tab_name):
        
        self.db_path = db_path
        self.filt_path_range = filt_path_range
        self.filt_frac_good = filt_frac_good
        self.tab_name = tab_name
        
        self.feats, self.experiments = get_feats_db(db_path, 
                         filt_path_range, 
                         filt_frac_good,
                         tab_name)
    
    def plot(self, feat_div, feat_ylim = [], valid_rows = None):
        
        if valid_rows is None:
            feats_f = self.feats
        else:
            feats_f = self.feats.loc[valid_rows]
        
        
        n_feat_div = len(feat_div)
        
        if n_feat_div <= 3:
            plt.figure()
            plot_boxes(feats_f, *feat_div)
            if feat_ylim:
                plt.ylim(feat_ylim)
        
        elif n_feat_div == 4:
            plot_boxes_group(feats_f, *feat_div, feat_ylim)   
        

if __name__ == '__main__':
    import os
    
    database_dir = '/Users/ajaver/OneDrive - Imperial College London/compare_strains_DB'
    tab_name = 'medians_split'
    filt_path_range = 0
    filt_frac_good = 0
        
    database_name = 'control_experiments_short_movies_new.db'
    #database_name = 'control_experiments_movies_2h.db'
    #database_name = 'control_experiments_Test_20161027.db'
    db_path = os.path.join(database_dir, database_name)
    
    
    db_obj = plot_db(db_path, filt_path_range, filt_frac_good, tab_name)
    #%%
    for feat in ['length', 'midbody_width', 'midbody_speed_pos']:
        db_obj.plot([feat,  'Strain', 'N_Worms', 'exp_name'])
    
    #db_obj.plot(['length',  'Strain', 'N_Worms', 'Picker'])
    #db_obj.plot(['midbody_width',  'Strain', 'N_Worms', 'Picker'])
    #db_obj.plot(['midbody_speed_pos',  'Strain', 'N_Worms', 'Picker'])
    