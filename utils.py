import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
from sklearn.utils import shuffle
from imgaug import augmenters as iaa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Flatten,Dense
from tensorflow.keras.optimizers import Adam

def get_name(file_path):
    return file_path.split('\\')[-1]

def import_data_info(path):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names = columns)
    #print(data.head())
    #print(data['Center'][0])
    data['Center'] = data['Center'].apply(get_name)
    return data

def balance_data(data, display=True):
    n_bins = 31
    samples_per_bin = 1000
    hist, bins = np.histogram(data['Steering'],n_bins)
    #print(bins)

    #if display:
        #center = (bins[:-1] + bins[1:]) * 0.5
        #print(center)
        #plt.bar(center, hist, width = 0.06)
        #plt.plot((-1,1), (samples_per_bin, samples_per_bin))
        #plt.show()

    remove_index_list = []
    for j in range(n_bins):
        bin_data_list = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j + 1]:
                bin_data_list.append(i)

        bin_data_list = shuffle(bin_data_list)
        bin_data_list = bin_data_list[samples_per_bin:]
        remove_index_list.extend(bin_data_list)
    print('Removed images : ', len(remove_index_list))
    data.drop(data.index[remove_index_list], inplace=True)
    print('Remaining images : ', len(data))

    if display:
        hist, bins = np.histogram(data['Steering'], n_bins)
        center = (bins[:-1] + bins[1:]) * 0.5
        #print(center)
        plt.bar(center, hist, width = 0.06)
        plt.plot((-1,1), (samples_per_bin, samples_per_bin))
        plt.show()

    return data

def load_data(path, data):
    images_path = []
    steerings = []

    for i in range(len(data)):
        indexed_data = data.iloc[i]
        #print(indexed_data)
        images_path.append(os.path.join(path, 'IMG', indexed_data[0]))
        steerings.append(float(indexed_data[3]))

    images_path = np.asarray(images_path)
    steerings = np.asarray(steerings)
    return images_path, steerings

def augment_img(img_path, steering):
    img = mpimg.imread(img_path)
    ## PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={'x' : (-0.1, 0.1), 'y' : (-0.1, 0.1)})
        img = pan.augment_image(img)
    ## ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)
    ## BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.4, 1.2))
        img = brightness.augment_image(img)
    ## FLIP
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
    return img, steering

def pre_processing(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200,66))
    img = img /255
    return img

def batch_generator(images_path, steering_list, batch_size, train_flag):
    while True:
        img_batch = []
        steering_batch = []

        for i in range(batch_size):
            index = random.randint(0, len(images_path) - 1)
            if train_flag:
                img, steering = augment_img(images_path[index], steering_list[index])
            else:
                img = mpimg.imread(images_path[index])
                steering = steering_list[index]

            img = pre_processing(img)
            img_batch.append(img)
            steering_batch.append(steering)

        yield (np.asarray(img_batch), np.asarray(steering_batch))

def create_model():

    model = Sequential()
    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(36, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001), loss='mse')

    return model
