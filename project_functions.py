import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import torch
from torch import nn
from torch.nn import functional as F
from skimage import morphology
from scipy.spatial import distance
import operator
from skimage import transform
from skimage.measure import label


# extract every frame from a video
def extract_frames(video_file_name):
    vid_cap = cv2.VideoCapture(video_file_name)
    if vid_cap.isOpened() is False:
        print("Couldn't open video, EXIT")
        return None
    frame_num = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames = np.zeros((frame_num, height, width, 3))  # array with all frames
    for i in range(frame_num):
        ret, frames[i, :, :, :] = vid_cap.read()
        if ret is False:
            print("error while reading frame ", i, ", exiting")
            return None

    return np.flip(frames.astype('uint8'), axis=-1)  # flip because R and B channels are inversed



# extract all objects and there center from the frames of the video (uses al the frames)
def get_operands(images, avoid_shaky_plus=False):
    # average over every frame and over the 3 channels to remove robot
    m_im = images.astype('long').mean(axis=0)
    m_im = m_im.mean(axis=2).astype('uint8')

    # gaussian adaptive threshold
    m_im = cv2.adaptiveThreshold(m_im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                cv2.THRESH_BINARY,151,20)

    # morphology: remove small holes
    m_im = morphology.remove_small_holes(m_im.astype(bool), 10)
    
    if avoid_shaky_plus:
        # do the same without averaging over all the frames, using only first frame
        m_im2 = images[0, :, :, :].copy()
        m_im2 = m_im2.mean(axis=2).astype('uint8')

        # gaussian adaptive threshold
        m_im2 = cv2.adaptiveThreshold(m_im2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,151,20)

        # morphology: remove small holes
        m_im2 = morphology.remove_small_holes(m_im2.astype(bool), 10)
        
        # use both to remove both shaky plus sign and robot
        m_im[m_im != m_im2] = True

    # re-convert to uint8 with  value from 0 to 255
    m_im = m_im.astype('uint8')
    m_im[m_im == 0] = 255
    # object labeling
    labeled_img, num_lab = label(m_im, background=0, return_num=True)
    
    # detect object close enough to be part of the same object (e.g. = or divide)
    center=np.zeros(2)
    center_comp=np.zeros(2)
    for i in range(1, num_lab+1):
        area_i = (labeled_img==(i)).sum()
        if area_i > 500:
            labeled_img[labeled_img==(i)] = 0
            continue
        for j in range(1, num_lab+1): 
            area_i = (labeled_img==(i)).sum()         
            area_j = (labeled_img==(j)).sum()
            if area_j > 500:
                labeled_img[labeled_img==(j)] = 0
                continue
            if i==j or area_i==0 or area_j==0:
                continue
            else:
                index = np.nonzero(labeled_img==i)
                center[0]=index[0].mean()
                center[1]=index[1].mean()
                index = np.nonzero(labeled_img==j)
                center_comp[0]=index[0].mean()
                center_comp[1]=index[1].mean()
                
                _distance = distance.euclidean(center, center_comp)
                
                if _distance < 15:
                    labeled_img[labeled_img==j] = i
        area_i = (labeled_img==(i)).sum()
        if area_i > 500 or area_i < 40:
            labeled_img[labeled_img == i] = 0
        
    # remove empty labels
    for i in range(1, num_lab):
            size = (labeled_img==(i)).sum()
            while size==0 and np.any(labeled_img>=i):
                for j in range(i, num_lab):
                    labeled_img[labeled_img==(j+1)] = j
                size = (labeled_img==(i)).sum()

    # extract mini-image of each object, as well as their center
    list_objects = []
    centers = np.zeros((labeled_img.max(),2))
    for i in range(labeled_img.max()):
        obj = labeled_img == (i+1)  # select only object with label i+1
        index = np.nonzero(obj)  # find the index of every pixel of the object
        centers[i, 1] = index[0].mean()
        centers[i, 0] = index[1].mean()
        left = index[1].min()  # - 10 #get the bounds of the index
        right = index[1].max()  # + 10
        top = index[0].min()  # - 10
        bottom = index[0].max()  # + 10
        list_objects.append(labeled_img[top:bottom+1, left:right+1])
    # pad the mini-image so that they have the same shape, use 28x28, shape of mnist

    # heights = np.zeros(len(list_objects), dtype='int')
    # widths = np.zeros(len(list_objects), dtype='int')
    # for i in range(len(list_objects)):
    #     heights[i] = list_objects[i].shape[0]
    #     widths[i] = list_objects[i].shape[1]
    height = 28  # heights.max()
    width = 28  # widths.max()
    all_objects = np.zeros((len(list_objects), height, width))
    for i in range(len(list_objects)):
        vert = height - list_objects[i].shape[0]
        horiz = width - list_objects[i].shape[1]
        if vert > 0 and vert % 2 == 0:
            if horiz > 0 and horiz % 2 == 0:
                all_objects[i, :, :] = np.pad(list_objects[i], ((int(vert/2), int(vert/2)), (int(horiz/2),int(horiz/2))), mode='constant')
            elif horiz > 0:
                all_objects[i, :, :] = np.pad(list_objects[i], ((int(vert/2), int(vert/2)), (int((horiz-1)/2), int((horiz+1)/2))), mode='constant')
            elif horiz == 0:
                all_objects[i, :, :] = np.pad(list_objects[i], ((int(vert/2), int(vert/2)), (0, 0)), mode='constant')
        elif vert > 0:
            if horiz > 0 and horiz%2==0:
                all_objects[i, :, :] = np.pad(list_objects[i], ((int((vert-1)/2), int((vert+1)/2)), (int(horiz/2), int(horiz/2))), mode='constant')
            elif horiz > 0:
                all_objects[i, :, :] = np.pad(list_objects[i], ((int((vert-1)/2), int((vert+1)/2)), (int((horiz-1)/2), int((horiz+1)/2))), mode='constant')
            elif horiz == 0:
                all_objects[i, :, :] = np.pad(list_objects[i], ((int((vert-1)/2), int((vert+1)/2)), (0,0)), mode='constant')
        elif vert == 0:
            if horiz > 0 and horiz%2==0:
                all_objects[i, :, :] = np.pad(list_objects[i], ((0, 0), (int(horiz/2), int(horiz/2))), mode='constant')
            elif horiz > 0:
                all_objects[i, :, :] = np.pad(list_objects[i], ((0, 0), (int((horiz-1)/2), int((horiz+1)/2))), mode='constant')
            elif horiz == 0:
                all_objects[i, :, :] = np.pad(list_objects[i], ((0, 0), (0, 0)), mode='constant')
    all_objects[all_objects != 0] = 255
    return all_objects, centers


def track_arrow(frames):

    height, width, layers, _ = frames.shape
    frames_box = []
    for i in range(frames.shape[0]):
        img = frames[i, :, :, :]
        bool_1 = np.logical_and(img[:, :, 2] < 100, img[:, :, 1] < 100)
        bool_2 = np.logical_and(bool_1, img[:, :, 0] > 100)
        x, y = np.where(bool_2)
        arrow = np.zeros((img.shape[0], img.shape[1]))
        arrow[x, y] = 255
        arrow = morphology.remove_small_objects(arrow.astype(bool), min_size=400).astype(np.uint8)
        contours, _ = cv2.findContours(arrow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rotrect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rotrect)
        box = np.int0(box)
        frames_box.append(box)
    frames_box = np.asarray(frames_box, dtype='int')
    return frames_box


# standard convnet
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


# This Function classifies operands based on fourier descriptors
def operand_classifier(img):
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)  # CV2 wants contour to be 255. probably should change
    contours = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    plus_desc = np.array([5.22110236e+00, 8.63863008e-03, 1.08649715e-02, 5.85660388e-04])
    minus_desc = np.array([1.98786094e+01, 1.41660093e-02, 2.88907168e+00, 1.44626175e-02])
    cross_desc = np.array([7.62104402e+01, 2.03015210e-01, 9.61817231e-01, 3.07777352e-02])
    if len(contours) == 2:
        return '='
    if len(contours) == 3:
        return '/'
    if len(contours) == 1:
        cnt_x = contours[0][:, 0, :][:, 1]
        cnt_y = contours[0][:, 0, :][:, 0]
        f_desc = np.absolute(np.fft.fft(cnt_x + cnt_y * 1j))
        f_desc = f_desc[1:5]/f_desc[5]
        if distance.euclidean(f_desc, plus_desc) < distance.euclidean(f_desc, minus_desc) and \
           distance.euclidean(f_desc, plus_desc) < distance.euclidean(f_desc, cross_desc):
            return '+'
        elif distance.euclidean(f_desc, minus_desc) < distance.euclidean(f_desc, plus_desc) and \
             distance.euclidean(f_desc, minus_desc) < distance.euclidean(f_desc, cross_desc):
            return '-'
        else:
            return '*'


# This Function interprets characters as actual operands
def get_operator(op):
    return {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.itruediv,
        }[op]


class TINet(nn.Module):
    def __init__(self, arms):
        super(TINet, self).__init__()
        self.arms = arms
        self.conv1 = nn.Conv2d(1, 40, kernel_size=5)
        self.conv2 = nn.Conv2d(40, 80, kernel_size=3)
        self.conv3 = nn.Conv2d(80, 120, kernel_size=5)
        self.fc1 = nn.Linear(120, 540)
        self.fc2 = nn.Linear(540, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        # self.fc5 = nn.Linear(100, 10)

    def forward(self, x):
        # rotate
        rots = []
        for i in range(self.arms):
            rots.append(x[:,:,:,:,i].clone())
            rots[i] = F.max_pool2d(F.relu(self.conv1(rots[i])), kernel_size=2, stride=2)
            rots[i] = F.max_pool2d(F.relu(self.conv2(rots[i])), kernel_size=2, stride=2)
            rots[i] = F.relu(self.conv3(rots[i]))
            rots[i] = F.relu(self.fc1(rots[i].view(-1,120))).unsqueeze(2)
        # cat
        x = torch.cat(rots, dim=2)
        x = F.max_pool2d(x.unsqueeze(1), kernel_size=(1,self.arms))
        x = x.squeeze(3)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = self.fc4(x)
        return x.squeeze()


def recognize_digit(digit):
    digit_input = np.zeros((1, 1, 28, 28, 12))
    digit_input[0, 0, :, :, 0] = digit[:, :]
    for j in range(12-1):
        digit_input[0, 0, :, :, j+1] = transform.rotate(digit_input[0, 0, :, :, 0], (j+1)*(360/12), preserve_range=True)
    digit_input = torch.from_numpy(digit_input).float()
    digit_input.sub_(digit_input.mean()).div_(digit_input.std())
    Net = TINet(12)
    Net.load_state_dict(torch.load('./trained_CNN_model/last.pt', map_location=torch.device('cpu')))
    # , map_location=torch.device
    output = Net(digit_input)
    corr_num = output.detach().numpy().argmax()
    return corr_num


def generate_video(images, centers, all_objects, arrow_boxes, video_output_path):
    n_frames = images.shape[0]
    n_objects = centers.shape[0]
    detected_objects = np.zeros((n_frames, 2))
    center_points = np.zeros((n_frames, 2), dtype='int')
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    color = (0, 0, 255)
    offset = 27
    c_list = [None] * n_frames  # Don't know if we will need this. for now for every scene it holds a character
    operand_chars = []  # Adds an operand string when used
    operands = []  # Adds an operand
    digits = []
    thickness = 1
    object_counter = 0  # Counts number of objects observed so far. At 1 it will be number and so on
    video = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc('H','F','Y','U'), 2, (images.shape[2], images.shape[1]), True)

    for i, img in zip(range(n_frames), images):
        center_points[i] = int(np.average([np.min(arrow_boxes[i, :, 0]), np.max(arrow_boxes[i, :, 0])])),\
                           int(np.average([np.min(arrow_boxes[i, :, 1]), np.max(arrow_boxes[i, :, 1])]))
        img_box = cv2.drawContours(img.copy(), [arrow_boxes[i]], 0, 128, 0)
        if i > 0:
            for k in range(i, 0, -1):
                img_box = cv2.line(img_box, (center_points[k - 1, 0], center_points[k - 1, 1]),\
                                   (center_points[k, 0], center_points[k, 1]), (0, 255, 0), thickness=2)

        for j in range(n_objects):
            if np.min(arrow_boxes[i, :, 0]) < centers[j, 0] < np.max(arrow_boxes[i, :, 0]) and \
                    np.min(arrow_boxes[i, :, 1]) < centers[j, 1] < np.max(arrow_boxes[i, :, 1]):
                current_center = centers[j, :]
                if not (np.array_equal(current_center, detected_objects[i - 1]) or\
                        np.array_equal(current_center, detected_objects[i - 2]) or\
                        np.array_equal(current_center, detected_objects[i - 3]) or\
                        np.array_equal(current_center, detected_objects[i - 4]) or\
                        np.array_equal(current_center, detected_objects[i - 5])):
                    detected_objects[i] = current_center
                    object_counter += 1
                    if object_counter % 2 == 1:
                        #  This is where the digits should be classified
                        c_list[i] = str(recognize_digit(all_objects[j]))
                        digits.append(recognize_digit(all_objects[j]))
                    else:
                        current_operand = operand_classifier(all_objects[j])  # Operand classification happens here
                        operand_chars.append(current_operand)
                        c_list[i] = current_operand  # The character of the operand to print on image
                        if current_operand != '=':
                            operands.append(get_operator(current_operand))
                            c_list[i] = current_operand  # Mapping of the operand
                        else:
                            result = calc_result(c_list)
                            c_list[i + 1] = str(result)
                        # operands[0](2,3)
        org = [550, 50]
        for k in range(0, i + 1, 1):
            if c_list[k] is not None:
                img_box = cv2.putText(img_box, c_list[k], (org[0], org[1]), font, fontScale, color, thickness,
                                      cv2.LINE_AA)
                org[1] += offset
        video.write(np.flip(img_box.astype('uint8'), axis=-1))
    video.release()
    cv2.destroyAllWindows()
    return c_list

# This function calculates the result of the final expression
def calc_result(list_ope):
    res=""
    for i in list_ope:
        if i == '=':
            break
        if i is not None:
            res = res + i
    result = eval(res)
    return result