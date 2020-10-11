# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:21:19 2019

@author: rafme

Based on: https://answers.opencv.org/question/200077/combine-several-videos-in-the-same-window-python/

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


def overlay_image_alpha(img, img_overlay, pos, alpha_mask = 1):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    
    Based on: https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv/14102014
    
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    #In case alpha mask wants to be used... not here
#    alpha = alpha_mask[y1o:y2o, x1o:x2o]
#    alpha_inv = 1.0 - alpha
    alpha = 1
    alpha_inv = 0

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])
        
    return img


plt.close('all')
cv2.destroyAllWindows()

#The directory where the file is taken
dn = os.path.dirname(os.path.realpath(__file__))


firstVideo = easygui.fileopenbox()  
secondVideo = easygui.fileopenbox()  

savedir = secondVideo.split(secondVideo.split('\\')[-1])[0]
savename = firstVideo.split('\\')[-1] + '_combined.avi'

#Necessary to write videos
fourcc = cv2.VideoWriter_fourcc(*'XVID')


# Read video
video1 = cv2.VideoCapture(firstVideo)
video2 = cv2.VideoCapture(secondVideo)




length1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
length2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))

#width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps1 = int(video1.get(cv2.CAP_PROP_FPS))
fps2 = int(video2.get(cv2.CAP_PROP_FPS))

if fps1 != fps2:
    raise("Both videos need to have same fps")
#if length1 != length2:
#    raise("Both videos need to have same length")


ret1, frame1 = video1.read()
ret2, frame2 = video2.read()


concatenate = False
overlap = True


if ret1 == True and ret2 == True: 
    
    if overlap:

        f = 0.5
        frame2resized = cv2.resize(frame2,(0,0),fx=f,fy=f)  
        both = overlay_image_alpha(frame1, frame2resized,
                                   (frame1.shape[1]-frame2resized.shape[1],0))

    elif concatenate:
        h1,w1,c1 = frame1.shape
        h2,w2,c2 = frame2.shape
        
        if h1 != h2 :
            scale = h1/h2
            dim = (int(w2*scale), h1)
            frame2 = cv2.resize(frame2,dim)     

        both = np.concatenate((frame1, frame2), axis=1)


    

pbar = tqdm(total=length1) #For profress bar


out = cv2.VideoWriter(savedir + savename + '.avi', cv2.VideoWriter_fourcc(*'XVID'), 
                              fps1, (both.shape[1],both.shape[0]))
out.write(both) #First frame written

while(video1.isOpened()):
    
    pbar.update() #Update progress bar

    ret1, frame1 = video1.read()
    ret2, frame2 = video2.read()
    
    if ret1 == True and ret2 == True: 
        
        if overlap:
    
            frame2resized = cv2.resize(frame2,(0,0),fx=f,fy=f)  
            both = overlay_image_alpha(frame1, frame2resized,
                                       (frame1.shape[1]-frame2resized.shape[1],0))

        elif concatenate:        
            h1,w1,c1 = frame1.shape
            h2,w2,c2 = frame2.shape
            
            if h1 != h2 :
                scale = h1/h2
                dim = (int(w2*scale), h1)
                frame2 = cv2.resize(frame2,dim)     
                
            both = np.concatenate((frame1, frame2), axis=1)
        
        
        out.write(both)
        
    else:
        break



out.release()
pbar.close() #Close progress bar







