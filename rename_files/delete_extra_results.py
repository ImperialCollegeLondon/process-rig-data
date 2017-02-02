#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 20:38:10 2016

@author: ajaver
"""

import os
import glob


root_dir = '/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests'

all_files = set(glob.glob(os.path.join(root_dir, '**', '*.hdf5'), recursive=True))


end_str = ['_skeletons.hdf5', '_features.hdf5', '_trajectories.hdf5', '_intensities.hdf5']



results_files = set(x for x in all_files if any(x.endswith(dd) for dd in end_str))
masked_files = all_files - results_files
assert all('/MaskedVideos/' in x for x in masked_files )


expected_results_files = set(x.replace('.hdf5', dd).replace('/MaskedVideos/', '/Results/') 
                            for x in masked_files for dd in end_str)

bad_files = results_files - expected_results_files

#for fname in bad_files:
#    os.remove(fname)