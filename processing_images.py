import numpy as np
import pandas as pd
import os, platform
import matplotlib.pyplot as plt
import cv2
import itertools
from multiprocessing import Pool, cpu_count

# derived code from:
# https://www.kaggle.com/kambarakun/intel-mobileodt-cervical-cancer-screening/how-to-start-with-python-on-colfax-cluster
def load_rgb_img(abspath_img):
    img = cv2.cvtColor(cv2.imread(abspath_img), cv2.COLOR_BGR2RGB)
    return img

def load_gry_img(abspath_img):
    img = cv2.cvtColor(cv2.imread(abspath_img), cv2.COLOR_BGR2GRAY)
    return img

def show_img(abspath_img):
    matplotlib.pyplot.imshow(sub_func_load_img(abspath_img))
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
        int_fill_1 = (480 - int_resize_1 ) // 2
        int_fill_2 =  480 - int_resize_1 - int_fill_1
        numpy_fill_1 = np.zeros((640, int_fill_1, 3),dtype=np.uint8)
        numpy_fill_2 = np.zeros((640, int_fill_2, 3), dtype=np.uint8)
        img_filled = np.concatenate((numpy_fill_1, img, numpy_fill_1), axis=1)

    elif img.shape[1] == 480:
        int_resize_0 = img.shape[0]
        int_fill_1 = (640 - int_resize_0 ) // 2
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

# worker function that does all the steps
def process_img(img_fn, rgb=True):
    if  not os.path.isfile(img_fn):
        print('this file doesnt exist!:\n--------------\n{}'.format(img_path))

    else:
        if rgb:
            img = load_rgb_img(img_fn)
        else:
            img = load_gry_img(img_fn)
        img = orient_img(img)
        img = resize_img_same_ratio(img)
        img = normalize_img(img)
        img = fill_img(img)
        return img

# not tested yet
def processing_helper(img_list):
    p = Pool(cpu_count())
    output = p.map(process_img, img_list) 
    img_array = np.array(output)
    p.close()
    p.join()
    return img_array

def test_function(img_path):
    # check if there is an error with the image
    if  not os.path.isfile(img_path):
        print('this file doesnt exist!:\n--------------\n{}'.format(img_path))

    else:
        img = load_rgb_img(img_path)
        plt.imshow(img)
        plt.show()

        plt.imshow(orient_img(img))
        plt.show()

def process_images_parallel(dirs_df_dir, fn):
    img_df = pd.read_csv(dirs_df_dir+fn+'.csv', header=0)
    img_paths = img_df['paths'].head()
    print('created df')
    arr = processing_helper(img_paths)
    return arr


def main():
    print(os.listdir())
    testing = False
    data_set = 'train'
    if 'c001' in platform.node():
        dirs_df_dir = '~/kaggle_code/data/'
        print('in colfax cluster')
        if testing:
            img_path = '/data/kaggle_3.27/test/5.jpg'
            test_function(img_path)
        else:
            files = ('train', 'test')
            for file in files:
                dir = '~/kaggle_code/data/'
                print('running {}'.format(file))
                arr = process_images_parallel(dir, file)
                print('saving fle')
                np.save(dir+file+'_processed',arr)

    else:
        print("What are you doing out of the cluster?!")


if __name__ == '__main__':
    main()
