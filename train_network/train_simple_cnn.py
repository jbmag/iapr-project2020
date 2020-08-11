import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torchvision import datasets
import matplotlib.pyplot as plt
import os
import some_functions as myfu
from skimage import transform
import random
train_rotated = True
siam = True
arms = 12
siam_rand = True
translation = True

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

print("Using device:{}".format(device))

dir_path = os.path.dirname(os.path.realpath(__file__))

images = np.zeros((10,28,28))
for i in range(10):
    images[i,:,:] = plt.imread(dir_path+'/extracted_images/symbol_{}.png'.format(i))

digits = images[[0,1,4,5,7,8],:,:]

if siam is True and siam_rand is False:
    print(1)
    train_input = torch.load('my_datasets/train_rot{}.pt'.format(arms))
    train_target = torch.load('my_datasets/train_tar_rot{}.pt'.format(arms))
    test_input = torch.load('my_datasets/test_rot{}.pt'.format(arms))
    test_target = torch.load('my_datasets/test_tar_rot{}.pt'.format(arms))
    print(train_input.shape, train_target.shape)
elif siam is True and siam_rand is True:
    print(2)
    train_input = torch.load('my_datasets/train_rot_rand{}.pt'.format(arms))
    train_target = torch.load('my_datasets/train_tar_rot_rand{}.pt'.format(arms))
    test_input = torch.load('my_datasets/test_rot_rand{}.pt'.format(arms))
    test_target = torch.load('my_datasets/test_tar_rot_rand{}.pt'.format(arms))
    print(train_input.shape, train_target.shape)
elif train_rotated is True:
    print(3)
    train_input = torch.load('my_datasets/train_rotn.pt')
    train_target = torch.load('my_datasets/train_tar_rotn.pt')
    test_input = torch.load('my_datasets/test_rotn.pt')
    test_target = torch.load('my_datasets/test_tar_rotn.pt')
    # train_input[-50:,0,:,:]=torch.from_numpy(digits[2,:,:])
    # train_target[-50:,:]=torch.Tensor([0,0,0,0,0,0,0,1,0,0])
    print(test_input.shape, test_target.shape)
elif train_rotated is False:
    print(4)
    train_input = torch.load('my_datasets/train.pt')
    train_target = torch.load('my_datasets/train_tar.pt')
    test_input = torch.load('my_datasets/test.pt')
    test_target = torch.load('my_datasets/test_tar.pt')
    print(test_input.shape, test_target.shape)
    
print(train_input.shape)



# transfer data to device used for training
train_input, train_target = train_input.to(device), train_target.to(device)
test_input, test_target = test_input.to(device), test_target.to(device)

#function to train a network (from lab3)
def train_model(model, train_input, train_target, test_input, test_target):
    criterion = nn.MSELoss() #use mean square error as loss function
    criterion.to(device)
    eta = 1e-1 #training rate
    mini_batch_size = 50 #batch size
    iterations = 40 #
    loss_vector = np.zeros(iterations)
    error_vector = np.zeros(iterations)
    error_vector_test = np.zeros(iterations)
    model = model.float()
    print("training...\n")
    for i in range(iterations):
        sum_loss = 0
        for b in range(0, train_input.shape[0], mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size)) #compute the output
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size)) #compute the loss
            with torch.no_grad():
                sum_loss = sum_loss + loss.item()
            model.zero_grad()
            loss.backward() #back-propagate the gradient
            with torch.no_grad():
                for p in model.parameters():
                    p -= eta * p.grad #update model parameters
        loss_vector[i] = sum_loss
        #compute error on train_set at this iteration
        # error_vector[i] = compute_errors(model, train_input, train_target) #evaluate the error on the train set
        error_vector_test = compute_errors_w_batch(model, test_input, test_target) #evaluate the error on the test set (not used for training, just info)
        print("Iteration ", i, ": loss = ", sum_loss, " , error on test: ", error_vector_test, "%")
    print("trained!")
    # return error_vector, error_vector_test, loss_vector

#function to compute classification error  (from lab3)
def compute_errors(model, test_input, test_target):
    #the component of the output with the highest activation is the determined class
    output = model(test_input)
    _, ind_max = output.max(axis = 1) #winner take all, the index of the maximum value of the output is the end class
    _, ind_max_tar = test_target.max(axis = 1)
    right_classifications = (ind_max == ind_max_tar).sum().item()
    error = ((test_input.shape[0] - right_classifications) / test_input.shape[0])*100
    return error

def compute_errors_w_batch(model, input, target, batch_size=25):
    error=0
    for b in range(0, input.size(0), batch_size):
        output = model(input.narrow(0, b, batch_size))
        _, ind_max = output.max(1)
        _, ind_max_tar = test_target.narrow(0, b, batch_size).max(axis = 1)
        right_classifications = (ind_max == ind_max_tar).sum().item()
        error += (batch_size - right_classifications) 
    return (error/input.shape[0])*100 #return a percentage

#train model
model = myfu.TINet(12)
model.load_state_dict(torch.load('./trained_CNN_model/my_model_rot_w_tran12.pt'))
model.to(device)
error_test = compute_errors_w_batch(model, test_input, test_target)
print("error on the testing set before training: {}%".format(error_test))
train_model(model, train_input, train_target, test_input, test_target)
error_test = compute_errors_w_batch(model, test_input, test_target)
# print("error on the testing set: {}%".format(error_test))

#export model
if train_rotated is not True:
    torch.save(model.state_dict(), './trained_CNN_model/my_model.pt')
    print("saved my_model.pt")
elif siam is True and siam_rand is False and translation is False:
    torch.save(model.state_dict(), './trained_CNN_model/my_model_rot{}.pt'.format(arms))
    print("saved my_model_rot{}.pt".format(arms))
elif siam is True and siam_rand is True and translation is False:
    torch.save(model.state_dict(), './trained_CNN_model/my_model_rot_rand{}.pt'.format(arms))
    print("saved my_model_rot_rand{}.pt".format(arms))
elif siam is True and siam_rand is True and translation is True:
    torch.save(model.state_dict(), './trained_CNN_model/my_model_rot_w_tran{}.pt'.format(arms))
    print("saved my_model_rot_w_tran{}.pt".format(arms))
else : 
    torch.save(model.state_dict(), './trained_CNN_model/my_model_rot_n.pt')
    print("saved my_model_rot_n.pt")
