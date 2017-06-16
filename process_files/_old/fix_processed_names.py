# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 16:15:39 2016

@author: worm_rig
"""

import os
import shutil
import glob
import numpy as np
import pandas as pd
import warnings
from functools import partial

if __name__ == '__main__':
    output_root = '/Volumes/behavgenom_archive$/Avelino/Worm_Rig_Tests/short_movies_new/'
    #'/Volumes/behavgenom_archive$/Avelino/PeterAskjaer/'
    exp_name = 'Double_pick_090217'#'Mutant_worm_screening_Y32H12A.7(ok3452)_220217'
    
    tsv_file = os.path.join(output_root, 'ExtraFiles', exp_name + '_renamed.tsv')
    
    tab = pd.read_table(tsv_file, names=['old', 'new'])
    
    
    for _, row in tab.iterrows():
        parts = row['old'].split(os.sep)
        delP = [int(x[2:]) for x in parts if x.startswith('PC')][0]
        old_base_name = os.path.splitext(os.path.basename(row['old']))[0]
        old_ch = [int(x[2:]) for x in old_base_name.split('_') if x.startswith('Ch')][0]
        
        base_name = os.path.splitext(os.path.basename(row['new']))[0]
        real_ch = 'Ch{}'.format(2*(delP-1)+old_ch)
        
        fparts = base_name.split('_')
        ff = [x.strip() if not x.startswith('Ch') else real_ch for x in fparts ]
        
        new_base_name = '_'.join(ff)
        
        search_str = os.path.join(output_root,'**', exp_name, base_name + '*')
        fnames = glob.glob(search_str)
        
        for bad_name in fnames:
            good_name = bad_name.replace(base_name, new_base_name)
            print(bad_name, good_name)
            #shutil.move(bad_name, good_name)
        
        





            


