#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:19:15 2016

@author: ajaver
"""

import os

from plot_db import plot_db
DATABASE_DIR = '/Users/ajaver/OneDrive - Imperial College London/compare_strains_DB'

tab_name = 'features_medians_split'
filt_path_range = 0
filt_frac_good = 0


database_name = 'control_experiments_short_movies.db'
db_path = os.path.join(DATABASE_DIR, database_name)
db_short_test = plot_db(db_path, filt_path_range, filt_frac_good, tab_name)