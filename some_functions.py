import cv2
import numpy as np
import matplotlib.pyplot as plt


#function to change illumination of the image, using the mean computed before
def normalize_image(im, means_ch):
    r_im = im.astype('int16')
    for nb_ch in range(3):
        r_im[:,:,nb_ch] = r_im[:,:,nb_ch] - (np.mean(r_im[:,:,nb_ch]) - means_ch[nb_ch])
        r_im[r_im[:,:,nb_ch] >= 256, nb_ch] = 255
        r_im[r_im[:,:,nb_ch] < 0, nb_ch] = 0
    r_im = r_im.astype('uint8')
    return r_im



def extract_frames(video_file_name):
    vid_cap = cv2.VideoCapture(video_file_name)
    if vid_cap.isOpened() is False:
        print("Couldn't open video, EXIT")
        return None
    frame_num = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames = np.zeros((frame_num, height, width, 3)) #array with all frames
    for i in range(frame_num):
       ret, frames[i, :, :, :] = vid_cap.read()
       if ret is False:
           print("error while reading frame ", i, ", exiting")
           return None

    return np.flip(frames.astype('uint8'), axis=-1) #flip because R and B channels are inversed 


def white_black_blue_red(im):
    white=np.array([255, 255, 255], dtype='float')
    black=np.array([0, 0, 0], dtype='float')
    blue=np.array([0, 0, 255], dtype='float')
    red=np.array([255, 0, 0], dtype='float')
    distance_to_color = np.zeros([im.shape[0], im.shape[1], 4], dtype='float')
    distance_to_color[:,:,0] = np.sqrt(((im.astype('float') - white)**2).sum(axis=2))
    distance_to_color[:,:,1] = np.sqrt(((im.astype('float') - black)**2).sum(axis=2))
    distance_to_color[:,:,2] = np.sqrt(((im.astype('float') - blue)**2).sum(axis=2))
    distance_to_color[:,:,3] = np.sqrt(((im.astype('float') - red)**2).sum(axis=2))
    ret = distance_to_color.argmin(axis=2)
    ret_im = np.zeros(im.shape)
    ret_im[ret==0,:]=255
    ret_im[ret==1,:]=0
    ret_im[ret==2,2]=255
    ret_im[ret==3,0]=255
    return ret_im.astype('uint8')

def find_arrow(image):
    #the arrow is the only red part
    arrow = image[:,:,0]
    arrow[np.logical_and(image[:,:,0] > 100, image[:,:,2] < 50)] = 255
    arrow[np.logical_or(image[:,:,0] <= 100, image[:,:,2] >= 50)] = 0
    plt.imshow(arrow)
    plt.show()


#separate each shape

#functions that returns a bool array with True values at the pixels a radius of size-pixels around pixel with coordinate (line, col)
def get_neighbours_coord(line, col, size, shape):
    neighb = np.zeros(shape, dtype=bool)
    neighb[(line-size):(line+size+1), (col-size):(col+size+1)] = True
    return neighb
    
    
#function to count object in a grey_scale image, with removed background
def count_objects(image):
    shape_num = 1
    label = np.zeros(image.shape)
    for lines in range(image.shape[0]):
        for cols in range(image.shape[1]):
            if image[lines, cols] == 255:
                continue
            neighb = get_neighbours_coord(lines, cols, 10, image.shape)
            if np.any(image[neighb] != 255):
                if label[neighb].max() == 0:
                    label[np.logical_and(image != 255, neighb)] = shape_num
                    shape_num = shape_num + 1
                else:
                    label[np.logical_and(image != 255, neighb)] = label[neighb].max()
    return label
                    