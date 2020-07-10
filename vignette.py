#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:11:01 2020

@author: haleyhunter-zinck
"""


from sehrd import sehrd
import numpy as np

file = '/Users/haleyhunter-zinck/Documents/workspace/synth/structured/output/nhamcs_2011.npy'
s = sehrd()
x = np.load(file)

s.train(x=x, method='medgan', n_epoch=3, batch_size=256)
#d = s.generate(obj=s, n_sample=1e3)