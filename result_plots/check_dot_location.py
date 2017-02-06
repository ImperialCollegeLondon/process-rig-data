#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 15:37:43 2016

@author: ajaver
"""

import os

from plot_db import plot_db
import numpy as np
import seaborn as sns
import pandas as pd
import tables

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.cmap'] = 'gray'

def plot_img_ch(exp_row, frame_number=0):
    mask_dir = exp_row['directory'].replace('Results', 'MaskedVideos')
    masks_file = os.path.join(mask_dir, exp_row['base_name'] + exp_row['ext'])
    skeletons_file = os.path.join(exp_row['directory'], exp_row['base_name'] + '_skeletons.hdf5')
    
    with tables.File(masks_file, 'r') as fid:
        img = fid.get_node('/full_data')[0]
    
    with pd.HDFStore(skeletons_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
                    
    
    ch2sp = {1:1, 2:4, 3:2, 4:5, 5:3, 6:6}
    coord = trajectories_data[trajectories_data['frame_number']==frame_number]    
    xx = coord['coord_x'].values
    yy = coord['coord_y'].values
    
    ch_n = exp_row['channel']
    
    ax = plt.subplot(2,3, ch2sp[ch_n], aspect='equal');
    plt.imshow(img);
    plt.scatter(xx, yy, s=80, facecolors='none', edgecolors='r')
    plt.axis('off');
    
    ax.text(0, 100, 'Ch%i' % ch_n, color='black', fontsize=8,
            bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':1})
    
    ax.text(0, 2000, exp_row['N_Worms'], color='white', fontsize=10,
            bbox={'facecolor':'green', 'alpha':0.5, 'pad':1})
    
    col = 'green' if exp_row['N_Worms'] == exp_row['n_worms_estimate'] else 'red'
    ax.text(1850, 2000, exp_row['n_worms_estimate'], color='white', fontsize=10,
            bbox={'facecolor':col, 'alpha':0.5, 'pad':1})
    
    if exp_row['Mark'] == 1:
        ax.text(1850, 100, 'XX', color='black', fontsize=8,
            bbox={'facecolor':'blue', 'alpha':1, 'pad':1})
        
if __name__ == '__main__':
    database_dir = '/Users/ajaver/OneDrive - Imperial College London/compare_strains_DB'
    tab_name = 'medians_split'
    filt_path_range = 0
    filt_frac_good = 0
     
    database_name = 'control_experiments_short_movies_new.db'
    db_path = os.path.join(database_dir, database_name)
    db_obj = plot_db(db_path, filt_path_range, filt_frac_good, tab_name)
    
    #%%
    
    sns.lmplot('N_Worms', 
                'n_worms_estimate', 
                hue = 'sample_id',
                data=db_obj.experiments,
                col='exp_name',
                col_wrap=2,
                x_jitter=0.25, 
                y_jitter=0.25, 
                legend=False,
                fit_reg=False)
#%%

#%%    
    #['double_pick_200117', 'double_pick_260117']
    for exp_name, exp_data in db_obj.experiments.groupby('exp_name'):
        print(exp_name)
        for set_n, set_data in exp_data.groupby('set_n'):
            for stage_pos, pos_data in set_data.groupby('stage_pos'):
                fig = plt.figure(figsize = (12, 8))
                title_str = 'S{} P{} {}'.format(set_n, stage_pos, exp_name)
                for _, row in pos_data.iterrows():
                    plot_img_ch(row)
                
                plt.subplots_adjust(wspace=0.01, hspace=0.01)
                fig.suptitle(title_str)
                
                print(title_str)
    #%%
#
#
#
#    #%%
#    #midbody_speed_pos #length
##    g = sns.lmplot(x="start_min", 
##               y="length", 
##               col="exp_name", 
##               hue="Strain", 
##               data=db_obj.feats,
##               col_wrap=2, 
##               size=3)   
##    g.set(yscale="log")
##    g.set(ylim=(500, 2000))
##    g.set(xlim=(-5, 625))
#
#    
#    #%%
#    if database_name == 'control_experiments_Agar_Test.db':
#        feat_name = "length"#"midbody_speed_pos"
#        for exp_name, exp_feats in db_obj.feats.groupby('exp_name'):
#            g = sns.lmplot(x="start_min", 
#                   y=feat_name, 
#                   col="Strain", 
#                   data=exp_feats,
#                   #hue='N_Worms',
#                   hue= 'Old/New Agar',
#                   col_wrap=2, 
#                   size=3)
#            g.set(ylim=(500, 1800))
#            g.set(xlim=(-5, 625))
#
#    #%%
#    
#    if False:
#        #%%
#        feat_name = "length"#"midbody_speed_pos"
#        for exp_name, exp_feats in db_obj.feats.groupby('exp_name'):
#            print(exp_name)
#            #if exp_name != 'Agar_Screening_181116': #'Agar_Screening_181116'
#            #    continue
#            
#            for strain, strain_feats in db_obj.feats.groupby('Strain'):
#                g = sns.lmplot(x="start_min", 
#                   y=feat_name, 
#                   col='sample_id',
#                   data=strain_feats,
#                   col_wrap=5, 
#                   size=3)
#                g.set(ylim=(500, 1800))
#                g.set(xlim=(-5, 625))
#            break
#        
#        #%%
#        import patsy
#        import statsmodels.api as sm
#        
#        f = 'length ~ start_min'
#        ols_fits = {}
#        for sample_id, strain_feats in db_obj.feats.groupby('sample_id'):
#            y,X = patsy.dmatrices(f, strain_feats, return_type='dataframe')
#            ols_fits[sample_id] = sm.OLS(y,X).fit()
#        #%%
#        sample_dat = db_obj.experiments.drop_duplicates('sample_id')
#        sample_dat.index = sample_dat['sample_id']
#        
#        sample_id,dat = zip(*((sample_id, (dat.params['Intercept'], dat.params['start_min'])) 
#                                    for sample_id, dat in ols_fits.items()))
#        fit_df = pd.DataFrame(np.array(dat), sample_id, ['Intercept', 'start_min'])
#        
#        sample_dat = sample_dat.join(fit_df)
#        
#        #%%
#        hue_feat = 'exp_name'#'N_Worms'#'Food_Conc'#'Old/New Agar'
#        g = sns.FacetGrid(sample_dat, col="Strain", hue=hue_feat, col_wrap=2)
#        g.map(plt.scatter, 'sample_id', 'start_min')
#        g.add_legend();
#        
#        g = sns.FacetGrid(sample_dat, col="Strain", hue=hue_feat, col_wrap=2)
#        g.map(plt.scatter, 'sample_id', 'Intercept')
#        g.add_legend();
#        #%%
#        x_feat = 'n_valid_skel' #'start_min' #'first_frame'#'n_frames' #
#        #y_feat = "midbody_speed_abs" 
#        y_feat = "length"#
#        
#        for strain, strain_feats in db_obj.feats.groupby('Strain'):
#            g = sns.FacetGrid(strain_feats, col_wrap=4, col="sample_id")
#            g.map(plt.scatter, x_feat, y_feat)
#            plt.subplots_adjust(top=0.9)
#            g.fig.suptitle(strain)
#    
    #%%
    #feat_name = "length"#"midbody_speed_pos"
    #g = sns.FacetGrid(db_obj.feats, row="Strain")
    #g.map(sns.coefplot, formula='length~start_min', groupby='sample_id')
    #%%
#    for exp_name, exp_feats in db_obj.feats.groupby('exp_name'):
#        print(exp_name)
#        #if exp_name != 'Agar_Screening_181116': #'Agar_Screening_181116'
#        #    continue
#        #plt.figure()
#        for ii, (strain, strain_feats) in enumerate(db_obj.feats.groupby('Strain')):
#            #plt.subplot(2,2, ii+1) 
#            g = sns.coefplot('length~start_min', strain_feats, groupby='sample_id')
#            plt.title(strain)
#            
#        break
    
    #%%
#               col_wrap=2, 
#
#        g.set(ylim=(500, 1800))
#        g.set(xlim=(-5, 625))
#        plt.title(strain)
        #break
    #%%
#    for exp_name, dat in experiments.groupby('exp_name'):
#        print('***************')
#        print(exp_name)
#        for bn in dat['base_name_d'].unique():
#            print(bn)