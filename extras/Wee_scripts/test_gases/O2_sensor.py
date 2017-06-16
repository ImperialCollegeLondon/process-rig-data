# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 17:31:27 2015

@author: ajaver
"""
import os
import csv, codecs
import matplotlib.pylab as plt
import numpy as np

root_dir = './29022016/'#'/Volumes/behavgenom$/Avelino/O2_test/'


for filename in os.listdir(root_dir):
    if filename[-4:] != '.csv':
        continue

    print(filename)
    data = []
    headers = []
    first_row = []
    with codecs.open(root_dir + filename, "r",encoding='utf-8', errors='ignore')  as csvfile:#open(fillename, 'r') as csvfile:
        next(csvfile) #skip the first line
        reader = csv.DictReader(csvfile, dialect='excel', delimiter = ';')
        
        
        field2save = 'Value'#'Phase'
        for row in reader:
            if not row[field2save] is None :
                if row[field2save] == '---':
                    dd = np.nan
                else:
                    dd = float(row[field2save])
                data.append(dd)
            if not headers:
                headers = row.keys()
                first_row = row
                
    plt.plot(data)
    
    np.savetxt(root_dir + filename.replace('.csv', '.txt'), np.array(data))
