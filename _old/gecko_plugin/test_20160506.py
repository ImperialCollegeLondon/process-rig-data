# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:00:29 2016

@author: ajaver
"""
import os
import glob
import tables
import numpy as np
import matplotlib.pylab as plt
#%%
main_dir = '/Volumes/D/test/20160607'

#files = os.listdir(main_dir)
files = glob.glob(os.path.join(main_dir, '*.hdf5'))

all_stamps = []
for fullname in files:
    fname = os.path.split(fullname)[1]
    
    try:
        with tables.File(fullname, 'r') as fid:
            #if fid.get_node('/mask').shape[0] > 5000:
            #    continue
            print(fname, fid.get_node('/mask'))
            all_stamps.append(fid.get_node('/stamp')[:])
    except tables.exceptions.HDF5ExtError:
        pass
        
#%%
for stamp in all_stamps:
    plt.figure()
    plt.plot(np.diff(stamp), '.')
    
#%%
#fname = '/Volumes/D/test/20160608_clock/Capture_Ch1_08062016_141140.hdf5'
#fname = '/Volumes/D/test/20160608_clock/Capture_Ch1_08062016_141058.hdf5'
fname = '/Volumes/D/test/20160607/Capture_Ch1_07062016_194853.hdf5'
#fname = '/Volumes/D/test/20160608_clock/Capture_Ch1_08062016_173003.hdf5'
with tables.File(fname, 'r') as fid:
    stamp = fid.get_node('/stamp')[:]

plt.figure()
plt.plot(np.diff(stamp), '.')
