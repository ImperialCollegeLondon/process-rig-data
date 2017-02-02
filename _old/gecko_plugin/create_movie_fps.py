# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:24:28 2016

@author: ajaver
"""

import cv2
import numpy as np


font = cv2.FONT_HERSHEY_SIMPLEX
fps = 50
fname = 'test_fps%i.avi' % fps

#plt.figure()
#plt.imshow(img, interpolation='none', cmap='gray')

im_size = (32, 256)
vid_writer = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc('M','J','P','G'), 
                         fps, im_size[::-1], isColor=True)

for frame in range(100000):
    if frame % 1000 == 0:
        print(frame)
    img = np.zeros((im_size[0], im_size[1], 3), np.uint8)
    cv2.putText(img,
    '%012i' % frame,(2,25), font, 1,(255,255,255), 2, cv2.LINE_AA)
    vid_writer.write(img)