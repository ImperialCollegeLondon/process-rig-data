#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:37:48 2017

@author: ajaver
"""
from itertools import combinations
import string
from random import shuffle

bad = 'DEUQ0124578'

dd = string.ascii_uppercase + string.digits
dd = [x for x in dd if x not in bad]
dd = [''.join(x) for x in combinations(dd, 3)]
dd = [x for x in dd if x[0] not in string.digits]

shuffle(dd)

with open('outputs.csv', 'w') as fid:
    dd = ['="{}"'.format(x) for x in dd]
    fid.write('\n'.join(dd))