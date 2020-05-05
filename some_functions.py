import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import torch
from torch import nn
from torch.nn import functional as F

#extract every frame from a video
def extract_frames(video_file_name):
    vid_cap = cv2.VideoCapture(video_file_name)
    if vid_cap.isOpened() is False:
        print("Couldn't open video, EXIT")
        return None
    frame_num = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames = np.zeros((frame_num, height, width, 3)) #array with all frames
    for i in range(frame_num):
       ret, frames[i, :, :, :] = vid_cap.read()
       if ret is False:
           print("error while reading frame ", i, ", exiting")
           return None

    return np.flip(frames.astype('uint8'), axis=-1) #flip because R and B channels are inversed 




#from Lab1:
#functions that returns a bool array with True values at the pixels a radius of size-pixels around pixel with coordinate (line, col)
def get_neighbours_coord(line, col, size, shape):
    neighb = np.zeros(shape, dtype=bool)
    neighb[(line-size):(line+size+1), (col-size):(col+size+1)] = True
    return neighb

#from Lab1:  
#function to count object in a grey_scale image, with removed background
def count_objects(image):
    shape_num = 1
    label = np.zeros(image.shape)
    for lines in range(image.shape[0]):
        for cols in range(image.shape[1]):
            if image[lines, cols] == 255:
                continue
            neighb = get_neighbours_coord(lines, cols, 10, image.shape)
            if np.any(image[neighb] != 255):
                if label[neighb].max() == 0:
                    label[np.logical_and(image != 255, neighb)] = shape_num
                    shape_num = shape_num + 1
                else:
                    label[np.logical_and(image != 255, neighb)] = label[neighb].max()
    return label
                    

#extract all objects and there center from the frames of the video (uses al the frames)
def get_operands(images, avoid_shaky_plus=False):
    #average over every frame and over the 3 channels to remove robot
    m_im = images.astype('long').mean(axis=0)
    m_im = m_im.mean(axis=2).astype('uint8')

    #gaussian adaptive threshold
    m_im=cv2.adaptiveThreshold(m_im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,151,20)

    #morphology: remove small holes
    m_im = morphology.remove_small_holes(m_im, 10)
    
    if avoid_shaky_plus:
        #do the same without averaging over all the frames, using only first frame
        m_im2 = images[0,:,:,:].copy()
        m_im2 = m_im2.mean(axis=2).astype('uint8')

        #gaussian adaptive threshold
        m_im2=cv2.adaptiveThreshold(m_im2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,151,20)

        #morphology: remove small holes
        m_im2 = morphology.remove_small_holes(m_im2, 10)
        
        #use both to remove both shaky plus sign and robot
        m_im[m_im!=m_im2]=True

    #re-convert to uint8 with  value from 0 to 255
    m_im = m_im.astype('uint8')
    m_im[m_im == 1] = 255

    #count objects with code from lab1
    label = count_objects(m_im).astype('uint8')
    
    #check num and size of object, remove big objects and their label
    for i in range(label.max()):
        size = (label==(i+1)).sum()
        while size >=700 or (size <40 and size >0):
            label[label==(i+1)] = 0
            for j in range(i+1, label.max()):
                label[label==(j+1)] = j
            size = (label==(i+1)).sum()
    
    #extract mini-image of each object, as well as their center
    list_objects = []
    centers = np.zeros((label.max(),2))
    for i in range(label.max()):
        obj = label==(i+1) #select only object with label i+1
        index = np.nonzero(obj) #find the index of every pixel of the object
        centers[i, 0] = index[0].mean()
        centers[i, 1] = index[1].mean()
        left = index[1].min() #- 10 #get the bounds of the index
        right = index[1].max() #+ 10
        top = index[0].min() #- 10
        bottom = index[0].max() #+ 10
        list_objects.append(label[top:bottom+1, left:right+1])
    
    #padd the mini-image so that they have the same shape, use 28x28, shape of mnist

    # heights = np.zeros(len(list_objects), dtype='int')
    # widths = np.zeros(len(list_objects), dtype='int')
    # for i in range(len(list_objects)):
    #     heights[i] = list_objects[i].shape[0]
    #     widths[i] = list_objects[i].shape[1]
    height = 28#heights.max()
    width = 28#widths.max()
    all_objects = np.zeros((len(list_objects), height, width))
    for i in range(len(list_objects)):
        vert = height - list_objects[i].shape[0]
        horiz = width - list_objects[i].shape[1]
        if vert > 0 and vert%2==0:
            if horiz > 0 and horiz%2==0:
                all_objects[i,:,:] = np.pad(list_objects[i], ((int(vert/2),int(vert/2)),(int(horiz/2),int(horiz/2))), mode = 'constant')
            elif horiz > 0:
                all_objects[i,:,:] = np.pad(list_objects[i], ((int(vert/2),int(vert/2)),(int((horiz-1)/2),int((horiz+1)/2))), mode = 'constant')
            elif horiz == 0:
                all_objects[i,:,:] = np.pad(list_objects[i], ((int(vert/2),int(vert/2)),(0,0)), mode = 'constant')
        elif vert > 0:
            if horiz > 0 and horiz%2==0:
                all_objects[i,:,:] = np.pad(list_objects[i], ((int((vert-1)/2),int((vert+1)/2)),(int(horiz/2),int(horiz/2))), mode = 'constant')
            elif horiz > 0:
                all_objects[i,:,:] = np.pad(list_objects[i], ((int((vert-1)/2),int((vert+1)/2)),(int((horiz-1)/2),int((horiz+1)/2))), mode = 'constant')
            elif horiz == 0:
                all_objects[i,:,:] = np.pad(list_objects[i], ((int((vert-1)/2),int((vert+1)/2)),(0,0)), mode = 'constant')
        elif vert == 0:
            if horiz > 0 and horiz%2==0:
                all_objects[i,:,:] = np.pad(list_objects[i], ((0,0),(int(horiz/2),int(horiz/2))), mode = 'constant')
            elif horiz > 0:
                all_objects[i,:,:] = np.pad(list_objects[i], ((0,0),(int((horiz-1)/2),int((horiz+1)/2))), mode = 'constant')
            elif horiz == 0:
                all_objects[i,:,:] = np.pad(list_objects[i], ((0,0),(0,0)), mode = 'constant')
    return all_objects, centers

#standard convnet
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x
 