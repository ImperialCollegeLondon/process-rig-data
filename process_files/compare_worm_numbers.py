#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:19:15 2016

@author: ajaver
"""
import os
import glob
import pandas as pd
import numpy as np

from create_results_db import get_rig_experiments_df
from view_dots_worms import get_n_worms_estimate

def _get_fname(row):
    return os.path.join(row['directory'],row['base_name'] + '_skeletons.hdf5')


if __name__ == '__main__':
    root_dir = '/Volumes/behavgenom_archive$/Avelino/screening/'
    exp_set = 'CeNDR' #'movies_2h'#'Agar_Test' #'Test_20161027' # 'L4_Long_Rec' #'Test_Food'#
    exp_set_dir = os.path.join(root_dir, exp_set)
    
    csv_dir = os.path.join(exp_set_dir, 'ExtraFiles')
    feats_dir = os.path.join(exp_set_dir, 'Results')
    
    
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv')) + glob.glob(os.path.join(csv_dir, '*.xlsx'))
    features_files = glob.glob(os.path.join(feats_dir, '**/*_features.hdf5'), recursive=True)
    
    experiments = get_rig_experiments_df(features_files, csv_files)
    
    
    all_worms = {}
    for strain, strain_data in experiments.groupby('Strain'):
        #only strains that are found in both days
        exp_group = strain_data.groupby('directory')
        if len(exp_group) > 1:
            for n_worms, dat in strain_data.groupby('N_Worms'):
                if len(dat) == 2:
                    print('**')
                    
                    field_name = '{}_W{}'.format(strain, n_worms)
                    assert not field_name in all_worms
                    all_worms[field_name] = np.full(2, np.nan)
                    
                    for ii, row in dat.iterrows():
                        skeletons_file = _get_fname(row)
                        ind_r = int('280417' in row['directory'])
                        
                        
                        with pd.HDFStore(skeletons_file, 'r') as fid:
                            trajectories_data = fid['/trajectories_data']
                        n_worms_estimate = get_n_worms_estimate(trajectories_data)
                        all_worms[field_name][ind_r] = n_worms_estimate
                        
                        print(ii, row['base_name'])
                        
#%%
for key in sorted(all_worms.keys()):
    print(key, all_worms[key])



