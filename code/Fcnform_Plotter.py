#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 10:48:54 2022

@author: kfair
"""

#Load in required libraries
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# Set working directory
home =  os.getcwd()[:-4]


# Define the expoential decay equation we're using for fitting
def decayfcn(x, A, b, k):
    y = A*np.exp(-b*(x**k))
    return y

# Define the data we want to plot
xdat = np.arange(0, 5, 0.1)

plt.figure(1, figsize=(10,5))

plt.subplot(121)
plt.plot(xdat, decayfcn(xdat,1,1,1), '-', label='A=1,    B=1,    C=1')
# plt.plot(xdat, decayfcn(xdat,0.5,0.5,1), '-', label='A=0.5, B=1,    C=1')
# plt.plot(xdat, decayfcn(xdat,1,0.5,1), '-', label='A=1,    B=0.5, C=1')
# plt.plot(xdat, decayfcn(xdat,1,0.5,2), '-', label='A=1,    B=0.5, C=2')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(title="Parameters")

plt.tight_layout()
plt.savefig('agedistfitter_demo1.png', bbox_inches="tight", dpi=500)
plt.show()

plt.figure(1, figsize=(10,5))

plt.subplot(121)
plt.plot(xdat, decayfcn(xdat,1,1,1), '-', label='A=1,    B=1,    C=1')
plt.plot(xdat, decayfcn(xdat,0.5,0.5,1), '-', label='A=0.5, B=1,    C=1')
# plt.plot(xdat, decayfcn(xdat,1,0.5,1), '-', label='A=1,    B=0.5, C=1')
# plt.plot(xdat, decayfcn(xdat,1,0.5,2), '-', label='A=1,    B=0.5, C=2')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(title="Parameters")

plt.tight_layout()
plt.savefig('agedistfitter_demo2.png', bbox_inches="tight", dpi=500)
plt.show()


plt.figure(1, figsize=(10,5))

plt.subplot(121)
plt.plot(xdat, decayfcn(xdat,1,1,1), '-', label='A=1,    B=1,    C=1')
plt.plot(xdat, decayfcn(xdat,0.5,0.5,1), '-', label='A=0.5, B=1,    C=1')
plt.plot(xdat, decayfcn(xdat,1,0.5,1), '-', label='A=1,    B=0.5, C=1')
# plt.plot(xdat, decayfcn(xdat,1,0.5,2), '-', label='A=1,    B=0.5, C=2')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(title="Parameters")

plt.tight_layout()
plt.savefig('agedistfitter_demo3.png', bbox_inches="tight", dpi=500)
plt.show()


plt.figure(1, figsize=(10,5))

plt.subplot(121)
plt.plot(xdat, decayfcn(xdat,1,1,1), '-', label='A=1,    B=1,    C=1')
plt.plot(xdat, decayfcn(xdat,0.5,0.5,1), '-', label='A=0.5, B=1,    C=1')
plt.plot(xdat, decayfcn(xdat,1,0.5,1), '-', label='A=1,    B=0.5, C=1')
plt.plot(xdat, decayfcn(xdat,1,0.5,2), '-', label='A=1,    B=0.5, C=2')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(title="Parameters")

plt.tight_layout()
plt.savefig('agedistfitter_demo4.png', bbox_inches="tight", dpi=500)
plt.show()
