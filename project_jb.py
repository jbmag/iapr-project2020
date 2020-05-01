import cv2
import numpy as np
import matplotlib.pyplot as plt
import some_functions as myfu
from skimage import morphology

#extract frames from video
im = myfu.extract_frames("robot_parcours_1.avi")
print(im.shape)
#extracts digits and operands from frames  
all_objects, centers = myfu.get_operands(im, avoid_shaky_plus=True)

# Plot images
fig, axes = plt.subplots(1, all_objects.shape[0])
for ax, i in zip(axes, all_objects):
    ax.imshow(i, cmap='gray')
    ax.axis('off')


print(centers)
print(im.shape)
fig,ax = plt.subplots()
ax.imshow(im[0,:,:,:])
ax.scatter(centers[:,1], centers[:,0])
plt.show()