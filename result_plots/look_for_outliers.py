#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:35:04 2016

@author: ajaver
"""

import matplotlib.pyplot as plt

from plot_db import get_feats_db
import os
import tables
from MWTracker.analysis.ske_create.helperIterROI import getWormROI

def plot_roi(feats, inds, n_rows = 5):
    plt.figure()
    for sub_ii, ind in enumerate(inds):
        row_db = feats.loc[ind]
        results_dir = row_db['directory']
        base_name = row_db['base_name']
        print(base_name)
        
        mask_dir = results_dir.replace('Results', 'MaskedVideos')
        
        masked_file = os.path.join(mask_dir, base_name + '.hdf5')
        skeletons_file = os.path.join(results_dir, base_name + '_skeletons.hdf5')
        
        
        with tables.File(skeletons_file, 'r') as skel_fid:
            tbl = skel_fid.get_node('/', 'trajectories_data')
            where_str = 'worm_index_joined == %i' % row_db['worm_index']
            
            for row_t in tbl.where(where_str):
                frame = row_t['frame_number']
                CMx = row_t['coord_x']
                CMy = row_t['coord_y']
                roi_size = row_t['roi_size']
                skeleton_id = row_t['skeleton_id']
                break
            
            skeletons = skel_fid.get_node('/', 'skeleton')[skeleton_id]
        
        with tables.File(masked_file, 'r') as mask_fid:
            img = mask_fid.get_node('/', 'mask')[frame]
            
        worm_roi, roi_corner = getWormROI(img, CMx, CMy, roi_size)
        skel_x = skeletons[:, 0] - roi_corner[0]
        skel_y = skeletons[:, 1] - roi_corner[1]
        
        
        plt.subplot(n_rows,n_rows, sub_ii+1)
        plt.imshow(worm_roi, cmap='gray', interpolation='none')
        plt.plot(skel_x, skel_y, 'r')
        plt.axis('off');
    plt.subplots_adjust(wspace=0.01, hspace=0.01)


if __name__ == '__main__':
    database_dir = '/Users/ajaver/OneDrive - Imperial College London/compare_strains_DB'
    tab_name = 'medians'
    filt_path_range = 0
    filt_frac_good = 0
    
    #database_name = 'control_experiments_Agar_Test.db'
    #database_name = 'control_experiments_Test_20161027.db'
    database_name = 'control_experiments_short_movies_new.db'
    
    feats, experiments = get_feats_db(os.path.join(database_dir, database_name), 
                         filt_path_range, 
                         filt_frac_good,
                         tab_name)
    
    #%%
    n_rows = 5
    tot_inds = n_rows*n_rows
    
    sorted_lengths = feats.length.sort_values()
    inds_top = sorted_lengths.index[-tot_inds:]
    plot_roi(feats, inds_top, n_rows)
    #%%
    inds_bot = sorted_lengths.index[:tot_inds]
    plot_roi(feats, inds_bot, n_rows)