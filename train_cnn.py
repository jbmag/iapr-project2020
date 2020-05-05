import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torchvision import datasets
import os
import some_functions as myfu

#download mnist data
data_dir = './data'
mnist_train_set = datasets.MNIST(data_dir + '/mnist/', train = True, download = True)
mnist_test_set = datasets.MNIST(data_dir + '/mnist/', train = False, download = True)

train_input = mnist_train_set.data.view(-1, 1, 28, 28).float()
#normalize data
mu, std = train_input.mean(), train_input.std()
train_input.sub_(mu).div_(std)

train_target = mnist_train_set.targets
#convert labels to one-hot
tmp = train_target.clone()
train_target = torch.zeros(train_input.shape[0], 10)
train_target[range(train_target.shape[0]), tmp] = 1


test_input = mnist_test_set.data.view(-1, 1, 28, 28).float()
#normalize test data
test_input.sub_(mu).div_(std)

test_target = mnist_test_set.targets
#convert labels to one-hot
tmp = test_target.clone()
test_target = torch.zeros(test_input.shape[0], 10)
test_target[range(test_target.shape[0]), tmp] = 1


#function to train a network (from lab3)
def train_model(model, train_input, train_target, test_input, test_target):
    criterion = nn.MSELoss() #use mean square error as loss function
    eta = 1e-1 #training rate
    mini_batch_size = 100 #batch size
    iterations = 25 #
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
        error_vector[i] = compute_errors(model, train_input, train_target) #evaluate the error on the train set
        error_vector_test[i] = compute_errors(model, test_input, test_target) #evaluate the error on the test set (not used for training, just info)
        print("Iteration ", i, ": loss = ", sum_loss, ", error = ", error_vector[i], ", error_test = ", error_vector_test[i])
    print("trained!")
    return error_vector, error_vector_test, loss_vector

#function to compute classification error  (from lab3)
def compute_errors(model, test_input, test_target):
    #the component of the output with the highest activation is the determined class
    output = model(test_input)
    _, ind_max = output.max(axis = 1) #winner take all, the index of the maximum value of the output is the end class
    _, ind_max_tar = test_target.max(axis = 1)
    right_classifications = (ind_max == ind_max_tar).sum().item()
    error = ((test_input.shape[0] - right_classifications) / test_input.shape[0])*100
    return error


#train model
model = myfu.Net()
error_vector, error_vector_test, loss_vector = train_model(model, train_input, train_target, test_input, test_target)

#export model
torch.save(model.state_dict(), './trained_CNN_model/my_model.pt')
