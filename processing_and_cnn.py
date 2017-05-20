
# coding: utf-8

# Save train and test images to normalized numpy arrays once for running multiple neural network configuration tests

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, platform, glob, itertools
from multiprocessing import Pool, cpu_count

from PIL import ImageFilter, ImageStat, Image, ImageDraw
from sklearn.preprocessing import LabelEncoder
import cv2

#--------------#

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization

from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras import optimizers
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras import backend as K


# In[ ]:

# derived code from:
# https://www.kaggle.com/kambarakun/intel-mobileodt-cervical-cancer-screening/how-to-start-with-python-on-colfax-cluster
# https://www.kaggle.com/the1owl/artificial-intelligence-for-cc-screening
'''
Processing functions
'''
def load_gry_img(fn):
    img = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2GRAY)
    return img

def show_img(fn):
    matplotlib.pyplot.imshow(sub_func_load_img(fn))
    matplotlib.pyplot.show()

# Orient images to be portriate
def orient_img(img):
    if img.shape[0] >= img.shape[1]:
        return img
    else:
        return np.rot90(img)

# make all images same size
def resize_img_same_ratio(img):
    if img.shape[0] / 640.0 >= img.shape[1] / 480.0:
        # (640, *, 3)
        img_resized = cv2.resize(img, (int(640.0 * img.shape[1] / img.shape[0]), 640)) 
    else:
        # (*, 480, 3)
        img_resized = cv2.resize(img, (480, int(480.0 * img.shape[0] / img.shape[1]))) 
    return img_resized

# fill in blank space with black
def fill_img(img):
    if img.shape[0] == 640:
        int_resize_1 = img.shape[1]
        int_fill_1 = (480 - int_resize_1 ) // 2 #floor
        int_fill_2 =  480 - int_resize_1 - int_fill_1
        numpy_fill_1 = np.zeros((640, int_fill_1, 3),dtype=np.uint8)
        numpy_fill_2 = np.zeros((640, int_fill_2, 3), dtype=np.uint8)
        img_filled = np.concatenate((numpy_fill_1, img, numpy_fill_1), axis=1)

    elif img.shape[1] == 480:
        int_resize_0 = img.shape[0]
        int_fill_1 = (640 - int_resize_0 ) // 2 #floor
        int_fill_2 = 640 - int_resize_0 - int_fill_1
        numpy_fill_1 = np.zeros((int_fill_1, 480, 3), dtype=np.uint8)
        numpy_fill_2 = np.zeros((int_fill_2, 480, 3), dtype=np.uint8)
        img_filled = np.concatenate((numpy_fill_1, img, numpy_fill_1), axis=0)

    else:
        raise ValueError

    return img_filled

# normalize pixel intesity to account for shadows and intesity variability within photo
def normalize_img(img):
    img_data = img.astype('float32')
    return img_data / 255  #255 comes from RBG format


# In[ ]:

''' 
input - filename
output - processed image

Reads image converts to RGB color
Flips image to portriat orientation
Resizes image to match orientation
Fills in blanks with black
Resizes image to input size based on arguement using bilinear interpolation
'''
def get_im_cv2(path, input_pic_dims = (64,64)):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = orient_img(img)
    img = resize_img_same_ratio(img)
    img = fill_img(img)
    img = resize_img_same_ratio(img)
    resized = cv2.resize(img, input_pic_dims, cv2.INTER_LINEAR)
    return [path, resized]

def normalize_image_features(paths):
    imf_d = {}
    p = Pool(cpu_count())
    ret = p.map(get_im_cv2, paths)
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[f] for f in paths]
    fdata = np.array(fdata, dtype=np.uint8)
    print(fdata.shape)
    
    fdata = fdata.transpose((0, 3, 1, 2))
    fdata = fdata.astype('float32')
    fdata = fdata / 255
    return fdata

def im_multi(path):
    try:
        im_stats_im_ = Image.open(path)
        return [path, {'size': im_stats_im_.size}]
    except:
        print(path)
        return [path, {'size': [0,0]}]

def im_stats(im_stats_df):
    im_stats_d = {}
    p = Pool(cpu_count())
    ret = p.map(im_multi, im_stats_df['path'])
    for i in range(len(ret)):
        im_stats_d[ret[i][0]] = ret[i][1]
    im_stats_df['size'] = im_stats_df['path'].map(lambda x: ' '.join(str(s) for s in im_stats_d[x]['size']))
    return im_stats_df


# In[ ]:

def run_processing(input_shape=(32,32)):
    computer = platform.node().lower()
    #   colfax cluster
    if 'c001' in computer: data_dir = '/data/kaggle/'
    
    #   local windows jupyter notebook
    elif 'shane' == computer: data_dir = 'C:/programming/CS674/kaggle_code/data/'
    
    #   local linux machine
    elif 'shane' in computer: data_dir = '~/programming/cs674/kaggle_code/data/
    
    #   kaggle kernel
    else: data_dir = '../input/'
    
    train = glob.glob(data_dir+'/train/**/*.jpg') #+ glob.glob(data_dir+'/additional/**/*.jpg')
    cols= ['type','image','path']
     #limit for Kaggle Demo
    train = pd.DataFrame([[p.split('/')[3],p.split('/')[4],p] for p in train], columns = cols)#[::3] 
    train = im_stats(train)
    train = train[train['size'] != '0 0'].reset_index(drop=True) #remove bad images
    train_data = normalize_image_features(train['path'])
    np.save('train.npy', train_data, allow_pickle=True, fix_imports=True)

    le = LabelEncoder()
    train_target = le.fit_transform(train['type'].values)
    print(le.classes_) #in case not 1 to 3 order
    np.save('train_target.npy', train_target, allow_pickle=True, fix_imports=True)

    test = glob.glob('../input/test/*.jpg')
    #[::20] #limit for Kaggle Demo
    test = pd.DataFrame([[p.split('/')[3],p] for p in test], columns = ['image','path'])#[::20] 
    test_data = normalize_image_features(test['path'])
    np.save('test.npy', test_data, allow_pickle=True, fix_imports=True)

    test_id = test.image.values
    np.save('test_id.npy', test_id, allow_pickle=True, fix_imports=True)


# In[ ]:

# run_processing()
# platform.node()
# os.listdir('~')
# os.listdir(os.path.dirname('~/programming/CS674/kaggle*/data/'))


# In[ ]:

def create_updated_model(opt='adamax', input_dims=(32,32,3)):
    model = Sequential()
    
    # C1
    # inputshape =(batch_size, steps, input_dim)  <- shape is inferred in consecutive layers
    model.add(Convolution2D(filters=4, kernel_size=3, strides=(1,1), input_shape=input_dims, padding='same'))
    model.add(BatchNormalization(axis=1, momentum=0.9, epsilon=0.001, center=True, scale=True))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))

    
    # C2
    model.add(Convolution2D(filters=4, kernel_size=3, strides=(1,1), input_shape=input_dims, padding='same'))
    model.add(BatchNormalization(axis=1, momentum=0.9, epsilon=0.001, center=True, scale=True))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='tf'))
    
    # C3
    model.add(Convolution2D(filters=4, kernel_size=3, activation='relu'))
    model.add(GlobalAveragePooling2D(data_format=None))
    
    # data_format = (batch, height, width, channels) <- shape is infered from previous layers
    # Using Average Pooling layer to instead of a fully connected layer ala GoogleNet, 
  
    model.add(Dense(12, activation='tanh'))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    return model


# In[ ]:

def create_base_model(opt= 'adamax', input_dims=(3,32,32)):
    model = Sequential()
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th', input_shape=input_dims))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(12, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    return model


# In[ ]:




# In[ ]:

def create_VGG_like_model():
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    model.fit(x_train, y_train, batch_size=32, epochs=10)
    score = model.evaluate(x_test, y_test, batch_size=32)


# In[ ]:

def run_model():
    # Setting state variables
    K.set_image_dim_ordering('th')
    K.set_floatx('float32')
    np.random.seed(17)

    # Reading in data
    train_data = np.load('train.npy')
    train_target = np.load('train_target.npy')

    # Cross fold training
    x_train,x_val_train,y_train,y_val_train = train_test_split(train_data,train_target,test_size=0.4, random_state=17)

    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    #datagen = ImageDataGenerator(rotation_range=0.3, zoom_range=0.3)

    datagen.fit(train_data)

    # Run Model
    model = create_model()
    model.fit_generator(datagen.flow(x_train,y_train, batch_size=15, shuffle=True), nb_epoch=200, samples_per_epoch=len(x_train), verbose=20, validation_data=(x_val_train, y_val_train))
    return model


# In[ ]:

def create_submission(model, fn='submission.csv'):
    # Load test data
    test_data = np.load('test.npy')
    test_id = np.load('test_id.npy')
    
    # create submission
    pred = model.predict_proba(test_data)
    df = pd.DataFrame(pred, columns=['Type_1','Type_2','Type_3'])
    df['image_name'] = test_id
    df = df[['image_name','Type_1','Type_2','Type_3']]
    df.to_csv(fn, index=False)
    
def print_submission(results = 'submission.csv'):
    with open(results) as fn:
        for line in fn:
            print(line)


# In[ ]:

def main():
    #----Model Params-----#
    batch_size = 32
    epochs = 200
    data_augmentation = True
    # most deep learning libraries run faster with input size of square 2^n
    dim = 2^6 #2^6 = 64
    input_shape = (dim, dim)
    
    
    full_run = True
    
    if full_run:
        run_processing()
        model = run_model()
        create_submission(model)
        print_submission()
    else:
        model = run_model()
        create_submission(model)


# In[ ]:

main()


# In[ ]:

print_submission()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



