#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:00:28 2016

@author: ajaver
"""

import glob
import os
import pandas as pd


main_dir = '/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests'
fnames = glob.glob(os.path.join(main_dir, '**', '*_features.hdf5'), recursive=True)


for ii, fname in enumerate(fnames):
    print('{} of {}'.format(ii+1, len(fnames)))
    with pd.HDFStore(fname, 'r') as fid:
        if not '/features_P90th_split' in fid:
            print(fname)
