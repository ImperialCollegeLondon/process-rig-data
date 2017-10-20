#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 13:44:05 2017

@author: ajaver
"""

import pandas as pd
import numpy as np

fname = '/Users/ajaver/Documents/GitHub/process-rig-data/tests/CeNDR/CeNDR_snps.csv'

snps = pd.read_csv(fname)

info_cols = snps.columns[:4]
strain_cols = snps.columns[4:]
snps_vec = snps[strain_cols].copy()
snps_vec[snps_vec.isnull()] = 0
snps_vec = snps_vec.astype(np.int8)


snps_c = snps[info_cols].join(snps_vec)

r_dtype = []
for col in snps_c:
    dat = snps_c[col]
    if dat.dtype == np.dtype('O'):
        n_s = dat.str.len().max()
        dt = np.dtype('S%i' % n_s)
    else:
        dt = dat.dtype
    r_dtype.append((col, dt))

snps_r = snps_c.to_records(index=False).astype(r_dtype)