# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:36:23 2016

@author: worm_rig
"""
import os
import sys
import glob
from functools import partial
import pandas as pd

from MWTracker.helperFunctions.runMultiCMD import runMultiCMD, print_cmd_list
from MWTracker.batchProcessing.ProcessMultipleFilesParser import TrackMultipleFilesParser
from MWTracker.batchProcessing.processMultipleFilesFun import trackMultipleFilesFun
from MWTracker.batchProcessing.CheckFinished import _checkFlagsFun

reformat_exec = os.path.join(os.path.split(__file__)[0], 'reformatMaskedVideo.py')

def get_file_parts(fname, videos_dir_root):
    parts = [x for x in fname.replace(videos_dir_root,'').split(os.sep) if x]
    pc_n = int(parts[0].replace('PC', ''))
    set_name, set_n, ch_str, date_str, time_str = \
                                parts[-1].rpartition('.')[0].split('_')
    assert(set_name == 'Set')
    set_n = int(set_n)
    ch_n = int(ch_str.replace('Ch', '')) + (pc_n-1)*2
    
    return (set_n, ch_n), (fname, date_str, time_str)

def processDBMaskedFiles(db_file, videos_dir_root, mask_dir_root):
    exp_name = os.path.splitext(os.path.split(db_file)[1])[0]
    save_dir = os.path.join(mask_dir_root, exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    db = pd.read_csv(db_file)
    video_plugin_files = glob.glob(os.path.join(videos_dir_root, 'PC*', 
                                                exp_name + '*',
                                                '**', 
                                                '*.hdf5'));
    
    #dictionary where the key is the set_n and ch_n
    g_fun = partial(get_file_parts, videos_dir_root = videos_dir_root)
    fparts = {x:y for x,y in map(g_fun, video_plugin_files)}
    
    cmd_list = []
    for ii, row in db.iterrows():
        ori_file, date_str, time_str = fparts[(row['Set_N'], row['Camera_N'])]
        fname = '{}_N{}_Set{}_Ch{}_Pos{}_{}_{}.hdf5'.format(
            row['Strain'],row['N_Worms'],row['Set_N'], row['Camera_N'], 
            row['Rig_Pos'], date_str, time_str)
        
        new_file = os.path.join(save_dir, fname) 
        
        
        
        if not _checkFlagsFun(new_file, '/mask', 1):      
            cmd_list.append((sys.executable, 
                             reformat_exec,
                             ori_file, 
                             new_file))
    
    
    return cmd_list

if __name__ == '__main__':
    max_num_process = 20

    #    db_file = r'D:\Experiments_CSV/FirstRun_181016.csv'
    videos_dir_root = r'D:\RawVideos'
    mask_dir_root = r'D:\MaskedVideos'
    results_dir_root = r'D:\Results'
    
    db_dir = r'D:\Experiments_CSV'
    
    cmd_list = []
    for db_file in glob.glob(os.path.join(db_dir, '*.csv')):
        print(db_file)
        cmd_list += processDBMaskedFiles(db_file, videos_dir_root, mask_dir_root)
    
    
    print_cmd_list(cmd_list)
    print(len(cmd_list))
    runMultiCMD(cmd_list, max_num_process = max_num_process) 

    dflt_vals = TrackMultipleFilesParser().dflt_vals
    dflt_vals['results_dir_root'] = results_dir_root
    dflt_vals['tmp_dir_root'] = ''
    dflt_vals['max_num_process'] = 15
    
#    trackMultipleFilesFun(mask_dir_root,**dflt_vals)

