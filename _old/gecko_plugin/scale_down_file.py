# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 17:40:41 2016

@author: ajaver
"""
import tables
import cv2
import matplotlib.pylab as plt
import os
import numpy as np

filters_tables = tables.Filters(complevel = 5, complib='zlib', shuffle=True)

h5_dir = '/Volumes/behavgenom$/Avelino/plugin_timestamps/'

#original_files = [x for x in os.listdir(h5_dir) if not 'Sampledown_' in x and 'Ch1' in x]
original_files = ['Capture_Ch1_09062016_151953.hdf5']
for fname in original_files:
    fname_new = 'Sampledown_' + fname
    print(fname)
    
    cols,rows = 64,64
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-44,1)
    with tables.File(os.path.join(h5_dir, fname), 'r') as fid, \
    tables.File(os.path.join(h5_dir, fname_new), 'w') as fid_new:
        original_images = fid.get_node('/mask')
        N = original_images.shape[0]
        stamp = fid.get_node('/stamp')[:]
        fid_new.create_array('/', 'stamp', obj = stamp)
        
        new_imgs = fid_new.create_carray('/', 'mask', shape = (N, cols, rows),\
        atom = tables.UInt8Atom(shape=()), filters = filters_tables)
        for ii in range(N):
            print(ii, N)
            im = fid.get_node('/mask')[ii]
            im_crop = cv2.resize(im, (cols,rows))
            im_crop = cv2.warpAffine(im_crop,M,(cols,rows))
            _, im_bw = cv2.threshold(im_crop,0,255, cv2.THRESH_OTSU)
            new_imgs[ii] = im_bw