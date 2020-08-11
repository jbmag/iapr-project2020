import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
from project_functions import TINet
import random
from skimage import transform
import gzip
import cv2

arms = 12       #number of arms in the siamese network, each correspond to one rotation of the mnist data

def extract_data(filename, image_shape, image_number):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(np.prod(image_shape) * image_number)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(image_number, image_shape[0], image_shape[1])
    return data


def extract_labels(filename, image_number):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * image_number)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


"""Function to generate randomly translated data from mnist, and convert the numpy arrays
    to torch tensors"""
def generate_rotated_mnist_for_siam_rand(train_input, train_label, size=60000):
    tmp = np.zeros((size,1,28,28,arms))
    labels = np.zeros((size,10))
    for i in range(size):
        l = random.randrange(0, train_input.shape[0], 1)
        labels[i,train_label[l]]=1
        T = np.float32([[1, 0, random.random() * 4 - 2], [0, 1, random.random() * 4 - 2]])
        tmp[i,0,:,:,0] = cv2.warpAffine(train_input[l,:,:],T,(28,28))
        for j in range(arms-1):
            tmp[i,0,:,:,j+1] = train_input[l,:,:]#transform.rotate(tmp[i,0,:,:,0], (j+1)*(360/arms), preserve_range=True)
    return torch.from_numpy(tmp).float(), torch.from_numpy(labels).float()


#extract data as numpy array
image_shape = (28, 28)
train_set_size = 60000
test_set_size = 10000
dir_path = os.getcwd()
data_part2_folder = os.path.join(dir_path, './data/mnist/MNIST/raw')

train_images_path = os.path.join(data_part2_folder, 'train-images-idx3-ubyte.gz')
train_labels_path = os.path.join(data_part2_folder, 'train-labels-idx1-ubyte.gz')
test_images_path = os.path.join(data_part2_folder, 't10k-images-idx3-ubyte.gz')
test_labels_path = os.path.join(data_part2_folder, 't10k-labels-idx1-ubyte.gz')

train_images = extract_data(train_images_path, image_shape, train_set_size)
test_images = extract_data(test_images_path, image_shape, test_set_size)
train_labels = extract_labels(train_labels_path, train_set_size)
test_labels = extract_labels(test_labels_path, test_set_size)

# COnvert data to torch tensor, with 'arms' rotations
train_input, train_target = generate_rotated_mnist_for_siam_rand(train_images, train_labels, 65000)
test_input, test_target = generate_rotated_mnist_for_siam_rand(test_images, test_labels, 10000)

#normalize data
mu, std = train_input.mean(), train_input.std()
train_input.sub_(mu).div_(std)
mu, std = test_input.mean(), test_input.std()
test_input.sub_(mu).div_(std)

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

print("Using device:{}".format(device))


# transfer data to device used for training
train_input, train_target = train_input.to(device), train_target.to(device)
test_input, test_target = test_input.to(device), test_target.to(device)

print("data is ready, moved to {}".format(device))

#function to train a network 
def train_model(model, train_input, train_target, test_input, test_target):
    criterion = nn.MSELoss() #use mean square error as loss function
    criterion.to(device)
    eta = 1e-1 #training rate
    mini_batch_size = 50 #batch size
    iterations = 30 
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
        error_vector_test = compute_errors_w_batch(model, test_input, test_target) #evaluate the error on the test set (not used for training, just info)
        print("Iteration ", i, ": loss = ", sum_loss, " , error on test: ", error_vector_test, "%")
    print("trained!")


#function to compute error on testing set, using mini batches
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
model = TINet(arms)
model.to(device)
error_test = compute_errors_w_batch(model, test_input, test_target)
print("error on the testing set before training: {}%".format(error_test))
train_model(model, train_input, train_target, test_input, test_target)
error_test = compute_errors_w_batch(model, test_input, test_target)

#save trained model to reuse in main project
torch.save(model.state_dict(), './trained_CNN_model/my_model_siam{}.pt'.format(arms))
print("saved my_model_siam{}.pt".format(arms))
