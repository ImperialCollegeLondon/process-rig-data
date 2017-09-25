#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 16:45:47 2017

@author: ajaver
"""

import os
import glob
import pandas as pd
import numpy as np
from functools import partial

from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.stats.multitest as smm
from scipy.stats import ttest_ind
import seaborn as sns
from misc import get_rig_experiments_df
import matplotlib.pylab as plt
from scipy.stats import f_oneway

index_cols = ['worm_index', 'timestamp', 'skeleton_id', 'motion_modes', 'strain']

def get_df_quantiles(input_d, f_ext):
    iexp, row = input_d
    print(iexp+1, row['base_name'])
    #%%
    fname = os.path.join(row['directory'], row['base_name'] + f_ext)
    with pd.HDFStore(fname, 'r') as fid:
        if '/timeseries_features' in fid:
            speed_cols = ['speed', 
                          'relative_radial_velocity_head_tip', 
                          'relative_radial_velocity_neck',
                          'relative_radial_velocity_hips',
                          'relative_radial_velocity_tail_tip',
                          'dist_from_food_edge',
                          'orientation_food_edge'
                          ]
            features_timeseries = fid['/timeseries_features']
        else:
            
            speed_cols = ['head_tip_speed', 'head_speed', 'midbody_speed', 'tail_speed', 'tail_tip_speed']
            features_timeseries = fid['/features_timeseries']
        
        
    #%%
    valid_feats = [x for x in features_timeseries.columns if not  x in index_cols]
    
    Q = features_timeseries[valid_feats].abs().quantile([0.1, 0.5, 0.9])
    Q.columns = [x+'_abs' for x in Q.columns]
    
    Q_s = features_timeseries[speed_cols].quantile([0.1, 0.5, 0.9])
    feat_mean = pd.concat((Q, Q_s), axis=1)
    
    #linearize the dataframe (convert to a pandas series series ) 
    dat = []
    for q, q_dat in feat_mean.iterrows():
        q_str = '_{}th'.format(int(round(q*100)))
        for feat, val in q_dat.iteritems():
            dat.append((val, feat+q_str))
    feat_mean_s = pd.Series(*list(zip(*dat)))
    
    return (iexp, feat_mean_s)


if __name__ == '__main__':
    exp_set_dir = '/Volumes/behavgenom_archive$/Avelino/screening/Swiss_Strains'
    
    csv_dir = os.path.join(exp_set_dir, 'ExtraFiles')
    feats_dir = os.path.join(exp_set_dir, 'Results')
        
    for f_ext in ['_featuresN.hdf5']:#, '_features.hdf5']:
        csv_files = glob.glob(os.path.join(csv_dir, '*.csv')) + glob.glob(os.path.join(csv_dir, '*.xlsx'))
        features_files = glob.glob(os.path.join(feats_dir, '**/*{}'.format(f_ext)), recursive=True)
        features_files = [x.replace(f_ext, '') for x in features_files]
        
        
        experiments = get_rig_experiments_df(features_files, csv_files)
        
         
        exp_inds, feat_means = zip(*map(partial(get_df_quantiles, f_ext=f_ext), experiments.iterrows()))
        #%%    
        feat_means_df = pd.concat(feat_means, axis=1).T
        feat_means_df.index = exp_inds
        feat_cols = feat_means_df.columns
        exp_cols = ['strain']
        feat_means_df = feat_means_df.join(experiments[exp_cols])
        #%%
        feat_strain_g = feat_means_df.groupby('strain')
        
        stats = []
        for feat in feat_means_df:
            if not feat in index_cols:
                dat = [g[feat].dropna().values for _, g in feat_strain_g]
                fstats, pvalue = f_oneway(*dat)
                stats.append((feat, fstats, pvalue))
        #%%
        feat, fstats, pvalue = zip(*stats)
        stat_values = pd.DataFrame(np.array((fstats, pvalue)).T, index=feat, columns=['fstat', 'pvalue'])
        
        #bonferroni correction
        stat_values['pvalue'] = stat_values['pvalue']*len(pvalue)
        stat_values = stat_values.sort_values(by='fstat', ascending = False)
        
        
        #%%
        median_values = feat_strain_g.median()
        
        
        save_name =  os.path.join('anova_{}.pdf'.format(f_ext))
        with PdfPages(save_name) as pdf_pages:
            for feat, row in stat_values.iterrows():
                dd = median_values[feat].argsort()
                strain_order = list(dd.index[dd.values])
                n2_ind = strain_order.index('N2')
                
                #%%
                f, ax = plt.subplots(figsize=(15, 6))
                sns.boxplot(x ='strain', y = feat, data = feat_means_df, order= strain_order, color="salmon"
                            )
                sns.swarmplot(x ='strain', y = feat, data = feat_means_df, color=".3", linewidth=0, order= strain_order)
                ax.xaxis.grid(True)
                ax.set(ylabel="")
                sns.despine(trim=True, left=True)
                plt.suptitle('{} (p-value {:.3})'.format(feat, row['pvalue']))
                plt.plot((n2_ind,n2_ind), plt.ylim(), '--k', linewidth=2)
                plt.xlabel('')
                #%%
                pdf_pages.savefig()
                plt.close(f)
            
            
            
