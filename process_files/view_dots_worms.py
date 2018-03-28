#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 15:37:43 2016

@author: ajaver
"""

import os

import numpy as np
import seaborn as sns
import pandas as pd
import tables
import glob

from scipy.misc import imresize

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.cmap'] = 'gray'

from matplotlib.backends.backend_pdf import PdfPages
from misc import get_rig_experiments_df

def get_n_worms_estimate(frame_numbers, percentile = 99):
    n_per_frame = frame_numbers.value_counts()
    n_per_frame = n_per_frame.values
    if len(n_per_frame) > 0:
        n_worms_estimate = np.percentile(n_per_frame, percentile)
    else:
        n_worms_estimate = 0
    return n_worms_estimate

def plot_img_ch(exp_row, frame_number=0, resize_factor = 0.25):
    mask_dir = exp_row['directory']
    mask_file = os.path.join(mask_dir, exp_row['base_name']) #exp_row['ext'])
    if not mask_file.endswith('.hdf5'):
        mask_file += '.hdf5'
    
    skeletons_file = mask_file.replace('MaskedVideos', 'Results').replace('.hdf5', '_skeletons.hdf5')
    
    with tables.File(mask_file, 'r') as fid:
        img = fid.get_node('/full_data')[0]
        tot_frames =  fid.get_node('/mask').shape[0]
        
        if resize_factor != 1:
            img = imresize(img, resize_factor)
        
    with tables.File(skeletons_file, 'r') as fid:
        if '/food_cnt_coord' in fid:
            food_cnt = fid.get_node('/food_cnt_coord')[:]*resize_factor
        else:
            food_cnt = None
    
    with pd.HDFStore(skeletons_file, 'r') as fid:
        trajectories_data = fid['/trajectories_data']
    
    trajectories_data = trajectories_data[trajectories_data['is_good_skel'] == 1]
    n_worms_estimate = get_n_worms_estimate(trajectories_data['frame_number'])
    
    ch2sp = {1:1, 2:4, 3:2, 4:5, 5:3, 6:6}
    coord = trajectories_data[trajectories_data['frame_number']==frame_number]    
    xx = coord['coord_x'].values*resize_factor
    yy = coord['coord_y'].values*resize_factor
    
    ch_n = exp_row['channel']
    
    ax = plt.subplot(2,3, ch2sp[ch_n], aspect='equal');
    plt.imshow(img);
    if food_cnt is not None:
        plt.plot(food_cnt[:, 0], food_cnt[:, 1], 'g', linewidth=1)
    
    plt.scatter(xx, yy, s=80, facecolors='none', edgecolors='r')
    plt.axis('off');
    
    ax.text(0, 100*resize_factor, 'Ch%i' % ch_n, color='black', fontsize=8,
            bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':1})
    ax.text(512*resize_factor, 100*resize_factor, 'Frames: {}'.format(tot_frames), 
            color='black', fontsize=8,
            bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':1})
    
    ax.text(0, 2000*resize_factor, exp_row['n_worms'], color='white', fontsize=10,
            bbox={'facecolor':'green', 'alpha':0.5, 'pad':1})
    
    col = 'green' if exp_row['n_worms'] == n_worms_estimate else 'red'
    ax.text(1850*resize_factor, 2000*resize_factor, n_worms_estimate, color='white', fontsize=10,
            bbox={'facecolor':col, 'alpha':0.5, 'pad':1})
    
    if exp_row['mark'] == 1:
        ax.text(1850*resize_factor, 100*resize_factor, 'XX', color='black', fontsize=8,
            bbox={'facecolor':'blue', 'alpha':1, 'pad':1})
    
    return n_worms_estimate

def _get_prefix_n(parts, prefix):
        good = [x for x in parts if prefix in x]
        assert len(good) == 1
        nn = int(good[0][len(prefix):])
        return nn
    
def _get_csv_fname(base_name):
    for ext in ['.csv', '.xlsx']:
        fname = base_name + ext
        if os.path.exists(fname):
            return fname
    return ''


def check_one_ch_per_set(exp_data):
    good = True
    for set_n, set_data in exp_data.groupby('set_n'):
        for set_del_t, delta_data in set_data.groupby('set_delta_time'):
            for stage_pos, pos_data in delta_data.groupby('stage_pos'):
                if len(pos_data.channel.unique()) != len(pos_data.channel):
                    
                    import pdb
                    pdb.set_trace()
                    msg = 'Set{} Pos{} T{} has more than one video per channel'.format(set_n, stage_pos, set_del_t) 
                    raise(ValueError(msg))
                
                
    return good

def make_plate_views(root_dir, exp_name, exp_data = None, max_del_t = 2, ignore_err=False):
    '''
    max_del_t -> max delta time in minutes consider movies as the same temporal set.
    '''

    if exp_data is None:
        csv_file = _get_csv_fname(os.path.join(root_dir, 'ExtraFiles', exp_name))
        search_str = os.path.join(root_dir, 'MaskedVideos', exp_name, '*.hdf5')
        mask_files = glob.glob(search_str)
        exp_data = get_rig_experiments_df(mask_files, [csv_file])
    
    if 'set_delta_time' in exp_data:
        exp_data['set_delta_time'] = np.floor(exp_data['set_delta_time']/max_del_t)*max_del_t
        check_one_ch_per_set(exp_data)
    
    exp_data['n_worms_estimate'] = np.nan
    figs2save = []
    for set_n, set_data in exp_data.groupby('set_n'):
        for set_del_t, delta_data in set_data.groupby('set_delta_time'):
            for stage_pos, pos_data in delta_data.groupby('stage_pos'):
                
                fig = plt.figure(figsize = (12, 8))
                title_str = 'Set{} Pos{} T{}min'.format(set_n, stage_pos, set_del_t)
                
                for irow, row in pos_data.iterrows():
                    try:
                        n_worms_estimate = plot_img_ch(row)
                    except:
                        if ignore_err:
                            continue
                        else:
                            raise
                    
                    exp_data.loc[irow, 'n_worms_estimate'] = n_worms_estimate
                    
                plt.subplots_adjust(wspace=0.01, hspace=0.01)
                fig.suptitle(title_str)
                figs2save.append(fig)
                
                print(exp_name, title_str)
                
    #plot correlation plots
    dd = exp_data[['n_worms_estimate', 'n_worms']].dropna()
    bot = min(dd['n_worms_estimate'].min(), dd['n_worms'].min())-1
    top = min(dd['n_worms_estimate'].max(), dd['n_worms'].max())+1
    
    obj = sns.lmplot('n_worms', 
                'n_worms_estimate',
                hue='set_n',
                data=exp_data,
                x_jitter=0.25, 
                y_jitter=0.25, 
                legend=False,
                fit_reg=False)
    
    plt.xlim((bot,top))
    plt.ylim((bot,top))
    
    figs2save = [obj.fig] + figs2save
    
    pdf_dir = os.path.join(root_dir, 'PlateViews')
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    save_name =  os.path.join(pdf_dir, exp_name + '_plate_view.pdf')
    pdf_pages = PdfPages(save_name)
    
    for fig in figs2save:
        pdf_pages.savefig(fig)
        
    pdf_pages.close()

def _plate_views_with_csv(dname):
    exp_names = [x for x in os.listdir(dname) if os.path.isdir(os.path.join(dname,x))]
    for exp_name in exp_names:
        make_plate_views(root_dir, exp_name, max_del_t=2)

#%%
if __name__ == '__main__':
    #root_dir = '/Volumes/behavgenom_archive$/Avelino/PeterAskjaer/'
    #exp_name = 'Mutant_worm_screening_Y32H12A.7(ok3452)_220217'
    #root_dir = '/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests/movies_2h/'
    #root_dir = '/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests/short_movies_new/'
    #root_dir = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR/'
    root_dir = '/Volumes/behavgenom_archive$/Serena/AggregationScreening/'
    #root_dir = '/Volumes/behavgenom_archive$/Adam/screening/antipsychotics'
    dname = os.path.join(root_dir, 'MaskedVideos')
    
    data_file = os.path.join(root_dir, 'aggregation_data_edit.csv')
    df_info = pd.read_csv(data_file)
    df_info = df_info.rename(columns={'basename':'base_name', 
                                      'dirname':'directory',
                                      'run':'set_n',
                                      'Dot':'mark'}, copy=False)
    df_info['set_delta_time'] = 0
    df_info['stage_pos'] = 0
    df_info['n_worms'] = 0
    
    #remove invalid data
    df_info = df_info[df_info['block']>0]
    df_info = df_info[~df_info['base_name'].str.lower().str.contains('_x_')]
    #%%
    for (block, day), dat in df_info.groupby(('block', 'day')):
        exp_name = '{}.{}'.format(block, day)
        print(exp_name)
        make_plate_views(root_dir, exp_name, exp_data = dat, ignore_err=True)
    
    
    #_plate_views_with_csv(dname)
    
        
    