import cv2
import numpy as np
import matplotlib.pyplot as plt
import some_functions as myfu
from skimage import morphology

#extract frames from video
im = myfu.extract_frames("robot_parcours_1.avi")

#average over every frame and over the 3 channels to remove robot
test = im.astype('long').mean(axis=0)
test = test.mean(axis=2).astype('uint8')

#gaussian adaptive threshold
test=cv2.adaptiveThreshold(test,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,151,20)

#morphology: remove small holes
kernel = np.ones((3,3),np.uint8)
test = morphology.remove_small_holes(test, 20)
f2=plt.figure(4)
plt.imshow(test, cmap='gray')


# #do the same without averaging over all the frames
# test2 = im[0,:,:,:].copy()
# test2 = test2.mean(axis=2).astype('uint8')

# #gaussian adaptive threshold
# test2=cv2.adaptiveThreshold(test2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,151,20)

# #morphology: remove small holes
# kernel = np.ones((3,3),np.uint8)
# test2 = morphology.remove_small_holes(test2, 20)
# f2=plt.figure(5)
# plt.imshow(test2, cmap='gray')
# plt.show()

# #remove both shaky plus sign and robot
# print(test.shape, test2.shape)
# test[test!=test2]=True
# plt.imshow(test, cmap='gray')
# plt.show()

#re-convert to uint8 with  value from 0 to 255
test = test.astype('uint8')
test[test == 1] = 255


#count objects with code from lab1
label = myfu.count_objects(test).astype('uint8')



#check num and size of object, remove big objects and their label
for i in range(label.max()):
    size = (label==(i+1)).sum()
    while size >=700:
        label[label==(i+1)] = 0
        for j in range(i+1, label.max()):
            label[label==(j+1)] = j
        size = (label==(i+1)).sum()



