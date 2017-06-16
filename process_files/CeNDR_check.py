#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 16:45:47 2017

@author: ajaver
"""

import os
import glob
import pandas as pd
import statsmodels.stats.multitest as smm
from scipy.stats import ttest_ind
import seaborn as sns
from misc import get_rig_experiments_df
import matplotlib.pylab as plt

def get_df_quantiles(input_d):
    iexp, row = input_d
    print(iexp, row['base_name'])
    
    fname = os.path.join(row['directory'], row['base_name'] + '_features.hdf5')
    with pd.HDFStore(fname, 'r') as fid:
        features_timeseries = fid['/features_timeseries']
    
    valid_feats = [x for x in features_timeseries.columns if not  x in index_cols]
    
    Q = features_timeseries[valid_feats].abs().quantile([0.1, 0.5, 0.9])
    Q.columns = [x+'_abs' for x in Q.columns]
    
    Q_s = features_timeseries[speed_cols].quantile([0.1, 0.5, 0.9])
    feat_mean = pd.concat((Q, Q_s), axis=1)
    
    #linearize the dataframe (convert to a pandas series series ) 
    dat = []
    for q, q_dat in feat_mean.iterrows():
        q_str = '{}th'.format(round(q*100))
        for feat, val in q_dat.iteritems():
            dat.append((val, feat+q_str))
    feat_mean_s = pd.Series(*list(zip(*dat)))
    
    return (iexp, feat_mean_s)


strain_list = pd.read_csv('strains_list.csv')
if __name__ == '__main__':
    
    
    exp_set_dir = '/Volumes/behavgenom_archive$/Avelino/screening/CeNDR'
    
    csv_dir = os.path.join(exp_set_dir, 'ExtraFiles')
    feats_dir = os.path.join(exp_set_dir, 'Results')
    
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv')) + glob.glob(os.path.join(csv_dir, '*.xlsx'))
    features_files = glob.glob(os.path.join(feats_dir, '**/*_features.hdf5'), recursive=True)
    experiments = get_rig_experiments_df(features_files, csv_files)
    
    control_strains = strain_list.loc[strain_list['set_type']=='divergent', 'strain'].values
    
    ctr_exp = experiments[experiments['strain'].isin(control_strains)]
    
    index_cols = ['worm_index', 'timestamp', 'skeleton_id', 'motion_modes']
    speed_cols = ['head_tip_speed', 'head_speed', 'midbody_speed', 'tail_speed', 'tail_tip_speed']
    
    exp_inds, feat_means = zip(*map(get_df_quantiles, ctr_exp.iterrows()))
    #%%    
    feat_means_df = pd.concat(feat_means, axis=1).T
    feat_means_df.index = exp_inds
    feat_str = feat_means_df.columns
    exp_cols = ['base_name', 'video_timestamp', 'exp_name', 'strain', 'n_worms']
    feat_means_df = feat_means_df.join(experiments[exp_cols])
    #%%
    
    
    #
    
    #%%
    strain_test = 'CB4856'
    strain_others = [x for x in control_strains if not x == strain_test]
    #%%
    feat_means_df_f = feat_means_df[feat_means_df['n_worms'] == 10]
    feat_strain_g = feat_means_df_f.groupby('strain')
    test_dat = feat_strain_g.get_group(strain_test)
    
    p_vals_l = []
    for ss in strain_others:
        pval_ss = []
        dat = feat_strain_g.get_group(ss)
        for feat in feat_str:
            x = test_dat[feat].dropna()
            y = dat[feat].dropna()
            if x.size ==0 or y.size ==0:
                continue
            _, p = ttest_ind(x, y)
            
            pval_ss.append((p, feat))
        pval_ss = pd.Series(*list(zip(*pval_ss)))
        p_vals_l.append((ss, pval_ss))
    #%%
    strains, dat = zip(*p_vals_l)
    pvals = pd.concat(dat, axis=1)
    pvals.columns=strains
    #%%
    
    pvals_corrected = {}
    for col in pvals:
        p = pvals[col].sort_values()
        p = p.dropna()
        
        reject, p_corr, alphacSidak, alphacBonf = \
                smm.multipletests(p.values, method = 'fdr_tsbky')
        pvals_corrected[col] = pd.Series(p_corr, p.index)
                
    #%%
    lab_order = sorted(control_strains)
    for ss in pvals_corrected:
        feat = pvals_corrected[ss].index[0]
        
        plt.figure()
        sns.stripplot(x='strain', 
                      y= feat, 
                      hue='n_worms', 
                      data=feat_means_df_f,
                      jitter=True,
                      order = lab_order)
        plt.title('{} p={:0.2}'.format(ss, pvals_corrected[ss][feat]))
    
    