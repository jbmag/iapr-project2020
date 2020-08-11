import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torchvision import datasets
import os
import some_functions as myfu
import random
from skimage import transform
import gzip
import cv2
import matplotlib.pyplot as plt
rotated = True
siam = True
arms=12
way =1
siam_rand = True

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


image_shape = (28, 28)
train_set_size = 60000
test_set_size = 10000
dir_path = os.getcwd()
data_part2_folder = os.path.join(dir_path, 'data/mnist/MNIST/raw')

train_images_path = os.path.join(data_part2_folder, 'train-images-idx3-ubyte.gz')
train_labels_path = os.path.join(data_part2_folder, 'train-labels-idx1-ubyte.gz')
test_images_path = os.path.join(data_part2_folder, 't10k-images-idx3-ubyte.gz')
test_labels_path = os.path.join(data_part2_folder, 't10k-labels-idx1-ubyte.gz')

train_images = extract_data(train_images_path, image_shape, train_set_size)
test_images = extract_data(test_images_path, image_shape, test_set_size)
train_labels = extract_labels(train_labels_path, train_set_size)
test_labels = extract_labels(test_labels_path, test_set_size)




print(train_images.shape)

def generate_rotated_mnist(train_input, train_label, size=60000):
    new_train = np.zeros((size, 1, 28, 28))
    new_label = np.zeros((size, 10))
    for i in range(size):
        l = random.randrange(0, train_input.shape[0], 1)
        k = random.random() * 360
        T = np.float32([[1, 0, random.random() * 4 - 2], [0, 1, random.random() * 4 - 2]])
        new_train[i,:,:,:] = transform.rotate(cv2.warpAffine(train_input[l,:,:],T,(28,28)), 
                                            random.random() * 360, preserve_range=True)
        new_label[i,train_label[l].item()] = 1
    return torch.from_numpy(new_train).float(), torch.from_numpy(new_label).float()


def generate_rotated_mnist_for_siam(train_input):
    tmp = np.zeros((train_input.shape[0],1,28,28,arms))
    tmp[:,0,:,:,0] = train_input
    for i in range(train_input.shape[0]):
        for j in range(arms-1):
            tmp[i,0,:,:,j+1] = transform.rotate(train_input[i,:,:], (j+1)*(360/arms), preserve_range=True)
    return torch.from_numpy(tmp).float()

def generate_rotated_mnist_for_siam_rand(train_input, train_label, size=60000):
    tmp = np.zeros((size,1,28,28,arms))
    labels = np.zeros((size,10))
    for i in range(size):
        l = random.randrange(0, train_input.shape[0], 1)
        labels[i,train_label[l]]=1
        T = np.float32([[1, 0, random.random() * 4 - 2], [0, 1, random.random() * 4 - 2]])
        # tmp[i,0,:,:,0] = transform.rotate(cv2.warpAffine(train_input[l,:,:],T,(28,28)), 
                                            # random.random() * 360, preserve_range=True)
        tmp[i,0,:,:,0] = cv2.warpAffine(train_input[l,:,:],T,(28,28))
        for j in range(arms-1):
            tmp[i,0,:,:,j+1] = transform.rotate(tmp[i,0,:,:,0], (j+1)*(360/arms), preserve_range=True)
    return torch.from_numpy(tmp).float(), torch.from_numpy(labels).float()

def generate_n_rotated_mnist(train_input, train_label, num=15):
    tmp = np.zeros((num*train_input.shape[0],1,28,28))
    labels = np.zeros((num*train_input.shape[0], 10))
    for i in range(train_input.shape[0]):
        for j in range(num):
            labels[i*num+j,train_label[i].item()] = 1
            tmp[i*num+j,0,:,:] = transform.rotate(train_input[i,:,:], random.random() * 360, preserve_range=True)
    return torch.from_numpy(tmp).float(), torch.from_numpy(labels).float()

num_train = 800000
num_test = 1000

if siam is True and siam_rand is False:
    train_input = generate_rotated_mnist_for_siam(train_images)
    test_input = generate_rotated_mnist_for_siam(test_images)

    tmp = torch.from_numpy(train_labels).unsqueeze(1)
    train_target = torch.zeros(train_input.shape[0], 10)
    train_target.scatter_(1, tmp.long(), 1)

    tmp = torch.from_numpy(test_labels).unsqueeze(1)
    test_target = torch.zeros(test_input.shape[0], 10)
    test_target.scatter_(1, tmp.long(), 1)
elif siam is True and siam_rand is True:
    train_input, train_target = generate_rotated_mnist_for_siam_rand(train_images, train_labels, 65000)
    test_input, test_target = generate_rotated_mnist_for_siam_rand(test_images, test_labels, 10000)
elif rotated is True and way == 1:
    print("rot mnist")
    train_input, train_target = generate_rotated_mnist(train_images, train_labels, num_train)
    test_input, test_target = generate_rotated_mnist(test_images, test_labels, num_test)
elif rotated is True and way == 2:
    train_input, train_target = generate_n_rotated_mnist(train_images, train_labels, 13)
    test_input, test_target = generate_n_rotated_mnist(test_images, test_labels, 13)
elif rotated is False:
    train_input = torch.from_numpy(train_images).unsqueeze(1)
    test_input = torch.from_numpy(test_images).unsqueeze(1)
    train_target = torch.zeros(train_input.shape[0], 10)
    train_target.scatter_(1, torch.from_numpy(train_labels).long().unsqueeze(1), 1)
    test_target = torch.zeros(test_input.shape[0], 10)
    test_target.scatter_(1, torch.from_numpy(test_labels).long().unsqueeze(1), 1)


#normalize data
mu, std = train_input.mean(), train_input.std()
train_input.sub_(mu).div_(std)
mu, std = test_input.mean(), test_input.std()
test_input.sub_(mu).div_(std)


if siam is True and siam_rand is False:
    torch.save(train_input, 'my_datasets/train_rot{}.pt'.format(arms))
    torch.save(train_target, 'my_datasets/train_tar_rot{}.pt'.format(arms))
    torch.save(test_input, 'my_datasets/test_rot{}.pt'.format(arms))
    torch.save(test_target, 'my_datasets/test_tar_rot{}.pt'.format(arms))
elif siam is True and siam_rand is True:
    torch.save(train_input, 'my_datasets/train_rot_rand{}.pt'.format(arms))
    torch.save(train_target, 'my_datasets/train_tar_rot_rand{}.pt'.format(arms))
    torch.save(test_input, 'my_datasets/test_rot_rand{}.pt'.format(arms))
    torch.save(test_target, 'my_datasets/test_tar_rot_rand{}.pt'.format(arms))
elif rotated is True:
    torch.save(train_input, 'my_datasets/train_rotn.pt')
    torch.save(train_target, 'my_datasets/train_tar_rotn.pt')
    torch.save(test_input, 'my_datasets/test_rotn.pt')
    torch.save(test_target, 'my_datasets/test_tar_rotn.pt')
elif rotated is False:
    torch.save(train_input, 'my_datasets/train.pt')
    torch.save(train_target, 'my_datasets/train_tar.pt')
    torch.save(test_input, 'my_datasets/test.pt')
    torch.save(test_target, 'my_datasets/test_tar.pt')


