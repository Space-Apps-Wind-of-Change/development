import cv2
import numpy as np
import os
from os.path import isfile, join

pathIn= './'
pathOut = 'video.avi'

fps = 15
frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

#for sorting the file names properly
files.sort(key = lambda x: x[5:-4])
files.sort()

frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

#for sorting the file names properly
files.sort(key = lambda x: x[5:-4])

for i in range(len(files)-1):
    # filename=pathIn + files[i]
    #reading each files
    img = cv2.imread('./' + files[i])
    height, width, layers = img.shape
    size = (width,height)
    
    #inserting the frames into an image array
    frame_array.append(img)

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MP4V'), fps, size)

for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])

out.release()