#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 11:16:23 2022

@author: kfair
"""

import os
import glob
import imageio
from pygifsicle import optimize

home =  os.getcwd()[:-4]

fig_dir = f"{home}/data/gifimages"
country="Egypt"

filenames = sorted(glob.glob(f"{home}/data/gifimages/*{country}*.png"))
filenames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

images = []
for file_name in filenames:
    images.append(imageio.imread(file_name))
imageio.mimsave(f"{home}/data/gifimages/{country}.gif", images, fps=1)

optimize(f"{home}/data/gifimages/{country}.gif")
