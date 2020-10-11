# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 14:23:58 2019

@author: rafme
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
import imutils #to rotate image

plt.close('all')
cv2.destroyAllWindows()

#The directory where the file is taken
dn = os.path.dirname(os.path.realpath(__file__))


filename = easygui.fileopenbox()  

savedir = filename.split(filename.split('\\')[-1])[0]
savename = filename.split('\\')[-1] + '_crop'

#Necessary to write videos
fourcc = cv2.VideoWriter_fourcc(*'XVID')


# Read video
video = cv2.VideoCapture(filename)

length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))


# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()
 
# Read first frame.
ok, initialFrame = video.read()

if not ok:
    print('Cannot read video file')
    sys.exit()
    


f = 0.3

frameResized = cv2.resize(initialFrame,(0,0),fx=f,fy=f)    
    
# Define an initial bounding box
bbox = (287, 23, 86, 320)

###bbox has four components:
#The first one is the x-coordinate of the upper left corner
#The second one is the y-coordinate of the upper left corner
#The third one is the x-coordinate of the lower right corner
#The four one is the y-coordinate of the lower right corner
 
# Uncomment the line below to select a different bounding box
bbox = cv2.selectROI(frameResized, False)

bbox = tuple([z/f for z in bbox]) #Resizes box to original file size
initialBbox = bbox
initialCrop = initialFrame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
initialCrop = cv2.cvtColor(initialCrop, cv2.COLOR_BGR2GRAY)
#initialFrameResizedGray = cv2.cvtColor(frameResized, cv2.COLOR_BGR2GRAY)

count = 0
#timeList = list()
#centers = list()
#errorCrop = list()
#movementIndex = list()

pbar = tqdm(total=length) #For profress bar

timeCrop = True
cut = 15

if timeCrop:
    savename = savename + 'Time'

rotated = True
if rotated:
    savename = savename + '_rotated'

out = cv2.VideoWriter(savedir + savename + '.avi', cv2.VideoWriter_fourcc(*'XVID'), 
                              fps, (initialCrop.shape[1],initialCrop.shape[0]))
cropFrame = initialFrame[int(initialBbox[1]):int(initialBbox[1]+initialBbox[3]), 
                          int(initialBbox[0]):int(initialBbox[0]+initialBbox[2])]
#If image is rotated
if rotated:
    cropFrame = imutils.rotate_bound(cropFrame, angle=180)

out.write(cropFrame)


while True:
    # Read a new frame
    pbar.update() #Update progress bar
    ok, frame = video.read()

    count += 1
    
    if timeCrop:
        t = count/fps
        if t > cut:
            break


    
    if not ok:
        print(ok)
        break
          
    cropFrame = frame[int(initialBbox[1]):int(initialBbox[1]+initialBbox[3]), 
                              int(initialBbox[0]):int(initialBbox[0]+initialBbox[2])]

    #If image is rotated
    if rotated:
        cropFrame = imutils.rotate_bound(cropFrame, angle=180)

    #Save
    out.write(cropFrame)
    

pbar.close() #Close progress bar

cv2.destroyAllWindows()
out.release()
video.release()






