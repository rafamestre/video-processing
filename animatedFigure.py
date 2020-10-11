# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:50:48 2019

@author: rafme

Based on https://towardsdatascience.com/how-to-create-animated-graphs-in-python-bb619cc2dec1

"""

import os
import sys
import numpy as np
import cv2
import time
import glob
import matplotlib
from matplotlib import pyplot as plt
import easygui
import scipy.ndimage
from tqdm import tqdm
from scipy.optimize import curve_fit
import peakutils
from bresenham import bresenham
import seaborn as sns
import matplotlib.animation as animation


sns.set_context("talk", font_scale=2, rc={"lines.linewidth": 2})
#sns.set_style("white")
sns.set_style("ticks")
sns.set_palette(sns.color_palette("GnBu_d", 4))

def animate(i):
    plt.cla()
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.xlim(np.min(t),np.max(t))
    plt.ylim(np.min(y),np.max(y))
    plt.ylim(0,250)
    plt.tight_layout()
    datay = y[:int(i+1)] #select data range
    datat = t[:int(i+1)]
    p = sns.lineplot(x=datat, y=datay, color="b")
    plt.plot(datat[-1],datay[-1],'ro',markersize=15)
#    p.tick_params(labelsize=17)
    plt.setp(p.lines,linewidth=5)
#    print(i)


plt.close('all')
cv2.destroyAllWindows()

#The directory where the file is taken
dn = os.path.dirname(os.path.realpath(__file__))


filepath = easygui.fileopenbox()  
data = np.transpose(np.loadtxt(filepath, skiprows=1))

savedir = filepath.split(filepath.split('\\')[-1])[0]

dataX = data[0]
dataY = data[1]

if filepath.split('\\')[-1] == "forceTime.txt":
    t = dataY
    y = dataX
    ylabel = 'Force ($\mu$N)'
    savename = 'ForceAnimated'
elif filepath.split('\\')[-1] == "displacement.txt":
    t = dataX
    y = dataY
    ylabel = 'Displacement ($\mu$m)'
    savename = 'DisplacementAnimated'
elif filepath.split('\\')[-1] == "movementIndex.txt":
    t = dataX
    y = dataY
    ylabel = 'Intensity of contractions (a.u.)'
    savename = 'MovementIndexAnimanted'


'''If time is cropped'''
cut =15
idx = np.abs(t-cut).argmin()
t = t[:idx]
y = y[:idx]

if t[0] == 0:
    sec = t[1]
else:
    sec = t[0]
fps = 1/sec



Writer = animation.writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

fig = plt.figure(figsize=(12,8))
#plt.plot(times[1:],movementIndex)
plt.xlabel('Time (s)')
plt.ylabel(ylabel)
plt.xlim(np.min(t),np.max(t))
plt.ylim(np.min(y),np.max(y))
plt.ylim(0,250)
plt.tight_layout()


ani = matplotlib.animation.FuncAnimation(fig, animate, frames=int(len(t)), repeat=False)

ani.save(savedir + savename + '.mp4', writer=writer)










