import cv2
import numpy as np
import matplotlib.pyplot as plt
import some_functions as myfu
from skimage import morphology

#extract frames from video
im = myfu.extract_frames("robot_parcours_1.avi")

#extracts digits and operands from frames  
all_objects = myfu.get_operands(im, avoid_shaky_plus=True)

# Plot images
fig, axes = plt.subplots(1, all_objects.shape[0])
for ax, im in zip(axes, all_objects):
    ax.imshow(im, cmap='gray')
    ax.axis('off')
plt.show()
