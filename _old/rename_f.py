#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 17:27:24 2016

@author: ajaver
"""

import glob
import os
import pandas as pd

GECKO_DT_FMT = '%d%m%Y_%H%M%S'
def focus2microns_per_pixel(focus):
    return -0.1937*(focus)+13.4377

def _db_ind(db):
    return {(row['Set_N'],row['Rig_Pos'],row['Camera_N']) : irow 
             for irow, row in db.iterrows()}    

def _rowdict(full_name, csv_db_ind, csv_db):
    rest, fname = os.path.split(full_name);
    exp_dir = os.path.split(rest)[1]
    
    dd = fname.split('_')
    time_str = dd[-2]
    date_str = dd[-3]
    pos_n = int([x[3:] for x in dd if x.startswith('Pos')][0])
    ch_n = int([x[2:] for x in dd if x.startswith('Ch')][0])
    set_n = int([x[3:] for x in dd if x.startswith('Set')][0])
    
    irow = csv_db_ind[exp_dir][(set_n, pos_n, ch_n)]
    row = csv_db[exp_dir].loc[irow]
    rowdict = row.to_dict()
    rowdict['microns_per_pixel'] = focus2microns_per_pixel(rowdict['Focus'])
    rowdict['full_name'] = full_name
    rowdict['video_timestamp'] = pd.to_datetime(date_str + '_' + time_str, format=GECKO_DT_FMT)
    rowdict['exp_id'] = exp_dir
    return rowdict

def get_path_df_control(csv_dir, main_dir):
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    csv_db = {os.path.basename(x)[:-4] : pd.read_csv(x) for x in csv_files}
    
    
    csv_db_ind = {x:_db_ind(csv_db[x]) for x in csv_db}
    
    
    
    fnames = glob.glob(os.path.join(main_dir, '**/*_features.hdf5'), recursive=True)
    
    experiments = pd.DataFrame(_rowdict(x,csv_db_ind, csv_db) for x in fnames)
    return experiments


if __name__ == '__main__':
    csv_dir = '/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests/Test_20161027/Experiments_CSV'
    main_dir = '/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests/Test_20161027/Results'
    experiments = get_path_df_control(csv_dir, main_dir)