import cv2
import numpy as np
import matplotlib.pyplot as plt
import some_functions as myfu
from skimage import morphology
from skimage import transform
from skimage import filters
import random
import os
import torch

arms=12


def compute_errors_w_batch(model, input, target, batch_size=25):
    error=0
    for b in range(0, input.size(0), batch_size):
        output = model(input.narrow(0, b, batch_size))
        _, ind_max = output.max(1)
        _, ind_max_tar = test_target.narrow(0, b, batch_size).max(axis = 1)
        right_classifications = (ind_max == ind_max_tar).sum().item()
        error += (batch_size - right_classifications) 
    return (error/input.shape[0])*100 #return a percentage

dir_path = os.path.dirname(os.path.realpath(__file__))

images = np.zeros((10,28,28))

for i in range(10):
    images[i,:,:] = plt.imread(dir_path+'/extracted_images/symbol_{}.png'.format(i))

digits = images[[0,1,4,5,7,8],:,:]

mlist=[2,3,7,7,2,3]
#create tensor of rotated digits only 1 of each
test_input = np.zeros((1,1,28,28,arms))
test_target = np.zeros((1,10))
for i in range(test_input.shape[0]):
    test_target[i,mlist[i]]=1
    test_input[i,0,:,:,0] = digits[i+5,:,:]
    for j in range(arms-1):
        test_input[i,0,:,:,j+1] = transform.rotate(test_input[i,0,:,:,0], (j+1)*(360/arms), preserve_range=True)
test_input = torch.from_numpy(test_input).float()
test_target = torch.from_numpy(test_target)

test_input.sub_(test_input.mean()).div_(test_input.std())

my_Net = myfu.TINet(arms)
my_Net.load_state_dict(torch.load('./trained_CNN_model/my_model_siam8.pt'))
output = my_Net(test_input)
_, maxInd = output.max(axis=0)
print("classes with TINet:                     ", maxInd.item())

numb = 1000
test_input = np.zeros((6*numb,1,28,28,arms))
test_target = np.zeros((6*numb,10))
for L in range(6):
    for i in range(numb):
        test_target[L*numb+i, mlist[L]]=1
        k = random.random() * 360
        test_input[L*numb+i,0,:,:,0] = transform.rotate(digits[L,:,:], k, preserve_range=True)
        for j in range(arms-1):
            test_input[L*numb+i,0,:,:,j+1] = transform.rotate(test_input[L*numb+i,0,:,:,0], (j+1)*(360/arms), preserve_range=True)    
test_input = torch.from_numpy(test_input).float()
test_target = torch.from_numpy(test_target)

test_input.sub_(test_input.mean()).div_(test_input.std())

my_Net = myfu.TINet(arms)
my_Net.load_state_dict(torch.load('./trained_CNN_model/last.pt'))
error = compute_errors_w_batch(my_Net, test_input, test_target, batch_size=100)
print("classes with TINet, 50 random rotations: error = {}%".format(error))

test_input = torch.load('my_datasets/test_rot12.pt')
test_target = torch.load('my_datasets/test_tar_rot12.pt')
error = compute_errors_w_batch(my_Net, test_input, test_target, batch_size=100)
print("classes with TINet, on test set: error = {}%".format(error))

## Test Simple network trained with rotated mnist
my_Net = myfu.Net()
my_Net.load_state_dict(torch.load('./trained_CNN_model/my_model_rot_n.pt'))
test_simple = torch.from_numpy(digits).unsqueeze(1).float()
test_simple.sub_(test_simple.mean()).div_(test_simple.std())
out = my_Net(test_simple)


_,maxind = out.max(1)
print("classes with simple net trained w/ rot: ",maxind)
