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
# def decayfcn(x, A, b, k):
#     y = A*np.exp(-b*(x**k))
#     return y

def decay_fcn(x, A, B, C):
    
    y = A*np.ones(len(x))
    y[k:] = A*np.exp(-B*((x[k:] - x[k])**C))
    
    return y


# Define the data we want to plot
xdat = np.arange(0, 10, 0.1)
k=0

plt.figure(1, figsize=(10,5))

plt.subplot(121)
plt.plot(xdat, decay_fcn(xdat,1,1,1), '-', label='A=1,    B=1,    C=1,    k=0')
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
plt.plot(xdat, decay_fcn(xdat,1,1,1), '-', label='A=1,    B=1,    C=1,    k=0')
plt.plot(xdat, decay_fcn(xdat,0.5,1,1), '-', label='A=0.5, B=0.5,    C=1,    k=0')
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
plt.plot(xdat, decay_fcn(xdat,1,1,1), '-', label='A=1,    B=1,    C=1,    k=0')
plt.plot(xdat, decay_fcn(xdat,0.5,1,1), '-', label='A=0.5, B=1,    C=1,    k=0')
plt.plot(xdat, decay_fcn(xdat,1,0.1,1), '-', label='A=1,    B=0.1, C=1,    k=0')
# plt.plot(xdat, decayfcn(xdat,1,0.5,2), '-', label='A=1,    B=0.5, C=2')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(title="Parameters")

plt.tight_layout()
plt.savefig('agedistfitter_demo3.png', bbox_inches="tight", dpi=500)
plt.show()


plt.figure(1, figsize=(10,5))

plt.subplot(121)
plt.plot(xdat, decay_fcn(xdat,1,1,1), '-', label='A=1,    B=1,    C=1,    k=0')
plt.plot(xdat, decay_fcn(xdat,0.5,1,1), '-', label='A=0.5, B=1,    C=1,    k=0')
plt.plot(xdat, decay_fcn(xdat,1,0.1,1), '-', label='A=1,    B=0.1, C=1,    k=0')
plt.plot(xdat, decay_fcn(xdat,1,0.1,2), '-', label='A=1,    B=0.1, C=2,    k=0')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(title="Parameters")

plt.tight_layout()
plt.savefig('agedistfitter_demo4.png', bbox_inches="tight", dpi=500)
plt.show()


plt.figure(1, figsize=(10,5))

plt.subplot(121)
plt.plot(xdat, decay_fcn(xdat,1,1,1), '-', label='A=1,    B=1,    C=1,    k=0')
plt.plot(xdat, decay_fcn(xdat,0.5,1,1), '-', label='A=0.5, B=1,    C=1,    k=0')
plt.plot(xdat, decay_fcn(xdat,1,0.1,1), '-', label='A=1,    B=0.1, C=1,    k=0')
plt.plot(xdat, decay_fcn(xdat,1,0.1,2), '-', label='A=1,    B=0.1, C=2,    k=0')
k=20
plt.plot(xdat, decay_fcn(xdat,1,0.1,2), '-', label='A=1,    B=0.1, C=2,    k=2')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(title="Parameters")

plt.tight_layout()
plt.savefig('agedistfitter_demo5.png', bbox_inches="tight", dpi=500)
plt.show()
