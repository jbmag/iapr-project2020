import cv2
import numpy as np
import matplotlib.pyplot as plt
import some_functions as myfu
from skimage import morphology
import torch

#extract frames from video
im = myfu.extract_frames("robot_parcours_1.avi")
print(im.shape)
#extracts digits and operands from frames  
all_objects, centers = myfu.get_operands(im, avoid_shaky_plus=True)

print(all_objects.shape)

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


my_Net = myfu.Net()
my_Net.load_state_dict(torch.load('./trained_CNN_model/my_model.pt'))
my_Net = my_Net.double()

listdig=[0,1,4,5,7,8]
for i in range(len(listdig)):
    test = all_objects[listdig[i],:,:].copy()
    test = torch.from_numpy(test).clone()
    test.sub_(test.mean()).div_(test.std())
    test = test.view((1,1,test.shape[0], test.shape[1]))
    output = my_Net(test)
    _, maxInd = output.max(axis=1)
    print(i, maxInd)
    

plt.show()