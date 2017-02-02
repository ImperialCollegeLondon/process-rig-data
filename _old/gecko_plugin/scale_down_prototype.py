# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:04:54 2016

@author: ajaver
"""
import tables
import numpy as np
import numpy
import matplotlib.pylab as plt

file_name = '/Users/ajaver/Desktop/Videos/Sampledown_Capture_Ch1_09062016_151953.hdf5'

#%% Get prototype numbers
with tables.File(file_name, 'r+') as fid:
    stamp = fid.get_node('/stamp')[:]
    
    im_bw = fid.get_node('/mask')
    numbers_base = np.zeros((20, 43, 28), np.uint8)
    for ii in range(10):
        numbers_base[2*ii] = im_bw[26 + ii*5, 10:53, 1:29]
    
    for ii in range(10):
        if ii <= 8:
            numbers_base[2*ii + 1] = (numbers_base[2*ii] + numbers_base[2*(ii+1)])
        else:
            numbers_base[2*ii + 1] = (numbers_base[2*ii] + numbers_base[0])
    
    if '/prototype_number' in fid:
        fid.remove_node('/prototype_number')
    fid.create_array('/', 'prototype_number', obj=numbers_base)
    
#%% 

with tables.File(file_name, 'r') as fid:
    masks = fid.get_node('/mask')[:].astype(np.double)
    numbers_base = fid.get_node('/prototype_number')[:].astype(np.double)
    
higher_img = masks[:, 10:53, 1:29]
lower_img = masks[:, 10:53, 36:]

plt.figure()
plt.subplot(2,2,1)
plt.imshow(higher_img[10])
plt.subplot(2,2,2)
plt.imshow(lower_img[2])
plt.subplot(2,2,3)
plt.imshow(higher_img[10]-numbers_base[14])
plt.subplot(2,2,4)
plt.imshow(lower_img[2]-numbers_base[14])

#%%

def get_best_match(num_array, numbers_base):
    dd = num_array[:,np.newaxis,:,:]-numbers_base[np.newaxis,:, :,:]
    D = np.mean(np.abs(dd), axis=(2,3))
    min_ind = np.argmin(D, axis=1);
    min_val = D[np.arange(D.shape[0]), min_ind]
    
    numbers = min_ind/2.
    return numbers, min_val

high_num, err_h = get_best_match(higher_img, numbers_base)
low_num, err_l = get_best_match(lower_img, numbers_base)

video_num = np.floor(high_num)*10 + low_num
video_num[25::25] = np.nan

#%%
plt.figure()
lab_t, = plt.plot(np.diff(stamp), label='timestamp')

dd = np.diff(video_num)
dd[dd<-50]= np.nan
lab_v, = plt.plot(dd*20, label='video number')

plt.xlim((0,12000))
plt.ylim((0, 550))
plt.legend(handles=[lab_t, lab_v])
plt.xlabel('frame number')
plt.ylabel('time difference (ms)')


plt.xlim((1800, 2000))
#%%
#plt.figure()
#plt.imshow(lower_img[1328]-numbers_base[-1])
#plt.figure()
#plt.imshow(higher_img[1328]-numbers_base[1])


