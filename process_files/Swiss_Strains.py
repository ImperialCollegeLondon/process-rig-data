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

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, SparsePCA, FastICA
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.stats.multitest as smm
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pylab as plt
from scipy.stats import f_oneway

from misc import get_rig_experiments_df

from tierpsy_features.features import timeseries_columns, ventral_signed_columns, morphology_columns
from tierpsy_features.events import get_events

index_cols = ['worm_index', 'timestamp', 'skeleton_id', 'motion_modes', 'strain']

def get_df_quantiles(df, ventral_side = None):
    q_vals = [0.1, 0.5, 0.9]
    iqr_limits = [0.25, 0.75]
    
    valid_q = q_vals + iqr_limits
    
    if not ventral_side in ['clockwise', 'anticlockwise']:
        Q = df[ventral_signed_columns].abs().quantile(valid_q)
        Q.columns = [x+'_abs' for x in Q.columns]
    
    vv = [x for x in timeseries_columns if not x in ventral_signed_columns]
    Q_s = df[vv].quantile(valid_q)
    feat_mean = pd.concat((Q, Q_s), axis=1)
    
    dat = []
    for q in q_vals:
        q_dat = feat_mean.loc[q]
        q_str = '_{}th'.format(int(round(q*100)))
        for feat, val in q_dat.iteritems():
            dat.append((val, feat+q_str))
    
    
    #IQR = feat_mean.loc[0.75] - feat_mean.loc[0.25]
    #dat += [(val, feat + '_IQR') for feat, val in IQR.iteritems()]
    
    feat_mean_s = pd.Series(*list(zip(*dat)))
    return feat_mean_s
       

def get_event_stats(event_durations):
    region_labels = {
            'motion_mode': {-1:'backward', 1:'forward', 0:'paused'}, 
            'food_region': {-1:'outside', 1:'edge', 0:'inside'}
            }
    
    tot_times = event_durations.groupby('event_type').agg({'duration':'sum'})['duration']
    event_g = event_durations.groupby(('event_type', 'region'))
    event_stats = []
    for event_type, region_dict in region_labels.items():
        for region_id, region_name in region_dict.items():
            stat_prefix = event_type + '_' + region_name
            try:
                dat = event_g.get_group((event_type, region_id))
                dat = dat['duration'].values
            except:
                dat = np.zeros(1)
    
            stat_name = stat_prefix + '_duration_50th'
            stat_val = np.nanmedian(dat)
            event_stats.append((stat_val, stat_name))
            
            stat_name = stat_prefix + '_fraction'
            stat_val = np.nansum(dat)/tot_times[event_type]
            event_stats.append((stat_val, stat_name))
            
    event_stats_s = pd.Series(*list(zip(*event_stats)))
    return event_stats_s

#    if is_normalized:
#        
#        median_length = features_timeseries.groupby('worm_index').agg({'length':'median'})
#        median_length_vec = features_timeseries['worm_index'].map(median_length['length'])
#        
#        feats2norm = [
#               'speed',
#               'relative_speed_midbody', 
#               'relative_radial_velocity_head_tip',
#               'relative_radial_velocity_neck',
#               'relative_radial_velocity_hips',
#               'relative_radial_velocity_tail_tip',
#               'head_tail_distance',
#               'major_axis', 
#               'minor_axis', 
#               'dist_from_food_edge'
#               ]
#    
#        for f in feats2norm:
#            features_timeseries[f] /= median_length_vec
#        
#        curv_feats = ['curvature_head',
#                       'curvature_hips', 
#                       'curvature_midbody', 
#                       'curvature_neck',
#                       'curvature_tail']
#        
#        
#        for f in curv_feats:
#            features_timeseries[f] *= median_length_vec
#        
#        #dd = {x : x + '_N' for x in feats2norm + curv_feats}
#        #features_timeseries.rename(columns = dd, inplace=True)
#       

if __name__ == '__main__':
    from tierpsy.helper.params import read_fps
    
    #exp_set_dir = '/Volumes/behavgenom_archive$/Avelino/screening/Swiss_Strains'
    exp_set_dir = '/Users/ajaver/OneDrive - Imperial College London/swiss_strains'
    csv_dir = os.path.join(exp_set_dir, 'ExtraFiles')
    feats_dir = os.path.join(exp_set_dir, 'Results')
    
    set_type = 'featuresN'
    #%%
    save_dir = './results_{}'.format(set_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #%%
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv')) + glob.glob(os.path.join(csv_dir, '*.xlsx'))
    
    f_ext = '_{}.hdf5'.format(set_type)
    features_files = glob.glob(os.path.join(feats_dir, '**/*{}'.format(f_ext)), recursive=True)
    features_files = [x.replace(f_ext, '') for x in features_files]
    
    experiments = get_rig_experiments_df(features_files, csv_files)
    
    
    all_data = []
    for iexp, row in experiments.iterrows():
        print(iexp+1, len(experiments))
        fname = os.path.join(row['directory'], row['base_name'] + f_ext)
        
        fps = read_fps(fname)
        with pd.HDFStore(fname, 'r') as fid:
            features_timeseries = fid['/timeseries_features']
        
        feat_mean_s = get_df_quantiles(features_timeseries)
        
        
        event_list = []
        for worm_ind, worm_data in features_timeseries.groupby('worm_index'):
            speed = worm_data['speed'].values
            dist_from_food_edge = worm_data['dist_from_food_edge'].values
            worm_length = worm_data['length'].median()
            
        
            df, durations_df = get_events(speed, dist_from_food_edge, fps, worm_length)
            
            df.index = worm_data.index
            durations_df['worm_index'] = worm_ind
            event_list.append((df, durations_df))
        
        event_vector, event_durations = zip(*event_list)
        event_vector = pd.concat(event_vector)
        event_durations = pd.concat(event_durations, ignore_index=True)
        event_durations = event_durations[event_durations.columns[::-1]]
        
        event_stats_s = get_event_stats(event_durations)
        #feat_mean_s = feat_mean_s.append(event_stats_s)
        
        all_data.append((iexp, feat_mean_s))

    exp_inds, feat_means = zip(*all_data)
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
    strain_order = ['N2', 'DAG356', 
                    'TR2171', 'DAG618', 
                    'DAG658', 'DAG680', 
                    'DAG515', 'DAG675', 
                    'DAG666', 'DAG667', 'DAG668', 'DAG676', 
                    'DAG677', 'DAG678', 'DAG679']
    cols_ind = [0, 0, 
                1, 1, 
                2, 2, 
                3, 3, 
                4, 4, 4, 4, 4, 4, 4]
    current_palette = sns.color_palette()
    cols = [current_palette[x] for x in cols_ind]
    assert len(strain_order) == len(cols)
    col_dict = {k:v for k,v in zip(strain_order, cols)}
    #%%
    if True:
        #median_values = feat_strain_g.median()
        
        save_name =  os.path.join('{}/boxplot_anova.pdf'.format(save_dir))
        with PdfPages(save_name) as pdf_pages:
            for feat, row in stat_values.iterrows():
                
                f, ax = plt.subplots(figsize=(15, 6))
                sns.boxplot(x ='strain', 
                            y = feat, 
                            data = feat_means_df, 
                            order= strain_order,
                            palette = col_dict
                            )
                sns.swarmplot(x ='strain', y = feat, data = feat_means_df, color=".3", linewidth=0, order= strain_order)
                ax.xaxis.grid(True)
                ax.set(ylabel="")
                sns.despine(trim=True, left=True)
                plt.suptitle('{} (p-value {:.3})'.format(feat, row['pvalue']))
                plt.xlabel('')
                
                #dd = median_values[feat].argsort()
                #strain_order = list(dd.index[dd.values])
                #n2_ind = strain_order.index('N2')
                #plt.plot((n2_ind,n2_ind), plt.ylim(), '--k', linewidth=2)
                
                pdf_pages.savefig()
                plt.close(f)
    
    #%%
    df = feat_means_df.dropna()
    
    valid_feats = [x for x in feat_means_df if x not in ['strain'] ]
    X = df[valid_feats].values.copy()
    
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min)/(x_max - x_min)
    
    #X = (X - np.mean(X, 0))/np.std(X, 0)
    #%%
    pca = PCA()
    X_pca = pca.fit_transform(X)
    
    pca_s = SparsePCA()
    X_pca_s = pca_s.fit_transform(X)
    #%%
    
    #for p in [5, 8, 10, 12, 15, 20, 30]:
    tsne = TSNE(n_components=2, 
                #perplexity = p,
                init='pca',
                verbose=1, 
                n_iter=10000
                )# random_state=0)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure()
    plt.plot(X_tsne[:, 0], X_tsne[:, 1], 'o')
    #%%
    from matplotlib.lines import Line2D
    import itertools
    
    marker_cycle = itertools.cycle(Line2D.filled_markers)
    mks = [next(marker_cycle) for _ in strain_order]
    
    save_name =  os.path.join('{}/clustering.pdf'.format(save_dir))
    with PdfPages(save_name) as pdf_pages:
        dat = {'t-SNE':X_tsne, 'PCA':X_pca, 'PCA_Sparse':X_pca_s}
        for k,Xp in dat.items():
            
            
            X_df = pd.DataFrame(Xp[:, 0:2], columns=['X1', 'X2'])
            X_df['strain'] = df['strain'].values
            
            g = sns.lmplot('X1', # Horizontal axis
               'X2', # Vertical axis
               data=X_df, # Data source
               fit_reg=False, # Don't fix a regression line
               hue = 'strain',
               hue_order = strain_order,
               palette = col_dict,
               size= 8,
               scatter_kws={"s": 100},
               legend=False,
               aspect = 1.2,
               markers = mks
               )
            
            box = g.ax.get_position() # get position of figure
            g.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height]) # resize position
    
            g.ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.title(k)
            pdf_pages.savefig()
            
        plt.close('all')        
    #%%
    
    strT = 'PCA'
    Xr = X_pca
    
    save_name =  os.path.join('{}/{}.pdf'.format(save_dir, strT))
    with PdfPages(save_name) as pdf_pages:
    
        plt.figure()
        plt.plot(np.cumsum(pca.explained_variance_ratio_), '.')
        plt.title('Explained Variance')
        pdf_pages.savefig()
        
        for n in range(10):
            df['p_dist'] = Xr[:, n]
            
            f, ax = plt.subplots(figsize=(15, 6))
            
            sns.boxplot(x = 'strain', 
                        y = 'p_dist', 
                        data = df, 
                        order= strain_order,
                        palette = col_dict
                        )
            sns.swarmplot(x = 'strain', 
                          y = 'p_dist', 
                          data = df,
                          color=".3", 
                          linewidth=0, 
                          order= strain_order
                          )
            
            plt.title('{}_{} var explained: {:.2}'.format(strT, n+1, pca.explained_variance_ratio_[n]))
            pdf_pages.savefig()
            
        plt.close('all')
    #%%
    
    strT = 'PCA_Sparse'
    Xr = X_pca_s
    
    save_name =  os.path.join('{}/{}.pdf'.format(save_dir, strT))
    with PdfPages(save_name) as pdf_pages:
        #http://www.tandfonline.com/doi/pdf/10.1198/106186006X113430?needAccess=true
        q, r = np.linalg.qr(X_pca_s)
        explained_variance = np.diag(r)**2
        explained_variance_ratio = explained_variance/np.sum(explained_variance)
        
        plt.figure()
        plt.plot(np.cumsum(explained_variance_ratio), '.')
        plt.title('Explained Variance')
        pdf_pages.savefig()
        
        
        for n in range(10):
            df['p_dist'] = Xr[:, n]
            #df['p_dist'] = np.linalg.norm((X_pca - n2_m)[:,:(n+1)], axis=1)
            
            f, ax = plt.subplots(figsize=(15, 6))
            
            sns.boxplot(x = 'strain', 
                        y = 'p_dist', 
                        data = df, 
                        order= strain_order,
                        palette = col_dict
                        )
            sns.swarmplot(x = 'strain', 
                          y = 'p_dist', 
                          data = df,
                          color=".3", 
                          linewidth=0, 
                          order= strain_order
                          )
            
            pca_s.components_[0, :]
            plt.title('{}_{} var explained: {:.2}'.format(strT, n+1, explained_variance_ratio[n]))
            pdf_pages.savefig()
        
        plt.close('all')
    #%%
    save_name =  os.path.join('{}/{}.txt'.format(save_dir, strT))
    with open(save_name, 'w') as fid:
        for n in range(10):
            vec = pca_s.components_[n, :]
            inds, = np.where(vec!=0)
            dd = [(valid_feats[ii],vec[ii]) for ii in inds]
            
            fid.write('***** PCA Sparse {} *****\n'.format(n+1))
            for feat in sorted(dd):
                fid.write('{} {:.3}\n'.format(*feat))
            fid.write('\n')
    