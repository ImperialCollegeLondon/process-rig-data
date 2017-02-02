#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 08:19:55 2016

@author: ajaver
"""

import os

from plot_db import get_feats_db, plot_boxes, plot_boxes_group

from plot_db import plot_db


DATABASE_DIR = '/Users/ajaver/OneDrive - Imperial College London/compare_strains_DB'
#%%
def first_test(tab_name, filt_path_range, filt_frac_good):
    
    database_name = 'control_experiments_Test_20161027.db'
    db_path = os.path.join(DATABASE_DIR, database_name)
    db_obj = plot_db(db_path, filt_path_range, filt_frac_good, tab_name)
    
    db_obj.plot(['length',  'Strain', 'Picker'])
    db_obj.plot(['midbody_speed_pos',  'Strain', 'Picker'])
    db_obj.plot(['midbody_speed_pos', 'N_Worms', 'Picker'])
    
    return db_obj
#%%
def food_test(tab_name, filt_path_range, filt_frac_good):
    
    database_name = 'control_experiments_Test_Food.db'
    db_path = os.path.join(DATABASE_DIR, database_name)
    db_obj = plot_db(db_path, filt_path_range, filt_frac_good, tab_name)
    db_obj.plot(['length',  'Strain', 'Picker'])
    db_obj.plot(['midbody_speed_pos',  'Strain', 'Food_Conc'], [])
    db_obj.plot(['midbody_speed_pos',  'Strain', 'Food_Conc'], [0, 500])

    return db_obj
#%%
def agar_test(tab_name, filt_path_range, filt_frac_good):
    database_name = 'control_experiments_Agar_Test.db'
    db_path = os.path.join(DATABASE_DIR, database_name)
    db_obj = plot_db(db_path, filt_path_range, filt_frac_good, tab_name)
    
    #% Agar_Screening_181116 seems to be wrongly labeled
    db_obj.plot(['midbody_speed_pos',  'Strain', 'Old/New Agar', 'exp_name'], [])
    db_obj.plot(['length',  'Strain', 'Old/New Agar', 'exp_name'], [])
    
    return db_obj
#%%    
def old_single_worm():
    database_name = 'control_single.db'
    db_path = os.path.join(DATABASE_DIR, database_name)
    db_obj = plot_db(db_path, filt_path_range, filt_frac_good, tab_name = 'features_means')
    
    #select worms that are none 
    chg = (db_obj.feats['gene']!=db_obj.feats['gene']) 
    db_obj.feats.loc[chg, 'gene'] = 'N2'
    db_obj.feats = db_obj.feats.sort_values(by='gene', ascending=False)
    db_obj.feats = db_obj.feats.rename(columns={'gene':'Strain'})
    
    db_obj.plot(['length',  'Strain'])
    db_obj.plot(['midbody_speed_pos',  'Strain'])
    return db_obj
    


#%%

if __name__ == '__main__':
    
    tab_name = 'features_medians_split'
    filt_path_range = 0
    filt_frac_good = 0
    
    #%%
    first_test(tab_name, filt_path_range, filt_frac_good)
    food_test(tab_name, filt_path_range, filt_frac_good)
    agar_test(tab_name, filt_path_range, filt_frac_good)
    old_single_worm()
    #%%
    database_name = 'control_experiments_short_movies.db'
    db_path = os.path.join(DATABASE_DIR, database_name)
    db_obj = plot_db(db_path, filt_path_range, filt_frac_good, tab_name)
    
    valid_rows = ~db_obj.feats['Vortex'].isnull()#db_obj.feats['exp_name'] == 'single_picking_011216'
    feat_str = ['midbody_speed_pos',  'Strain', 'Vortex', 'exp_name']
    db_obj.plot(feat_str, valid_rows=valid_rows)
    
    valid_rows = db_obj.feats['exp_name'].str.contains('double')
    valid_rows &= ~db_obj.feats['exp_name'].str.contains('after')
    valid_rows &= db_obj.feats['Vortex'] != 1
    
    feat_str = ['midbody_speed_pos',  'Food_Conc', 'Strain', 'exp_name']
    db_obj.plot(feat_str, valid_rows=valid_rows)
    
    #%%
    
    
    
    