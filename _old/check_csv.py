#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:06:50 2016

@author: ajaver
"""

import glob
import os

from correct_file_name import read_rig_csv_db

csv_dir = '/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests/Test_20161027/ExtraFiles'

for csv_f in glob.glob(os.path.join(csv_dir, '*.csv')):
    db, db_ind = read_rig_csv_db(csv_f)
    print(csv_f)
    print(db['Strain'].unique())
    print(db['Picker'].unique())