import platform
import os, pickle
import pandas as pd




def get_file_paths():
    if 'c001' in platform.node(): 
#   colfax cluster
        abspath_dataset_dir_train_1 = '/data/kaggle/train/Type_1'
        abspath_dataset_dir_train_2 = '/data/kaggle/train/Type_2'
        abspath_dataset_dir_train_3 = '/data/kaggle/train/Type_3'
        abspath_dataset_dir_test    = '/data/kaggle/test/'
        abspath_dataset_dir_add_1   = '/data/kaggle_3.27/additional/Type_1_v2'
        abspath_dataset_dir_add_2   = '/data/kaggle_3.27/additional/Type_2_v2'
        abspath_dataset_dir_add_3   = '/data/kaggle_3.27/additional/Type_3_v2'
#   local machine
    elif '.local' in platform.node():
        abspath_dataset_dir_train_1 = '/abspath/to/train/Type_1'
        abspath_dataset_dir_train_2 = '/abspath/to/train/Type_2'
        abspath_dataset_dir_train_3 = '/abspath/to/train/Type_3'
        abspath_dataset_dir_test    = '/abspath/to/test/'
        abspath_dataset_dir_add_1   = '/abspath/to/additional/Type_1'
        abspath_dataset_dir_add_2   = '/abspath/to/additional/Type_2'
        abspath_dataset_dir_add_3   = '/abspath/to/additional/Type_3'
    else:
#   kaggle kernel
        abspath_dataset_dir_train_1 = '/kaggle/input/train/Type_1'
        abspath_dataset_dir_train_2 = '/kaggle/input/train/Type_2'
        abspath_dataset_dir_train_3 = '/kaggle/input/train/Type_3'
        abspath_dataset_dir_test    = '/kaggle/input/test/'
        abspath_dataset_dir_add_1   = '/kaggle/input/additional/Type_1'
        abspath_dataset_dir_add_2   = '/kaggle/input/additional/Type_2'
        abspath_dataset_dir_add_3   = '/kaggle/input/additional/Type_3'

    return abspath_dataset_dir_train_1, abspath_dataset_dir_train_2, abspath_dataset_dir_train_3, abspath_dataset_dir_test, abspath_dataset_dir_add_1, abspath_dataset_dir_add_2, abspath_dataset_dir_add_3


def get_list_abspath_img(abspath_dataset_dir):
    list_abspath_img = []
    for str_name_file_or_dir in os.listdir(abspath_dataset_dir):
        if ('.jpg' in str_name_file_or_dir) == True:
            list_abspath_img.append(os.path.join(abspath_dataset_dir, str_name_file_or_dir))
    list_abspath_img.sort()
    return list_abspath_img

def save_dirs(dir, img_dirs, type='df'):
    if type == 'df':
        img_dirs.to_csv(dir, index=False)
    else:
        with open(dir, 'wb') as file:
            pickle.dump(dir_list, file)


def save_img_dirs():
#    dir = '~/kaggle_code/'
#   Location of the colfax cluster data
    dir = os.path.join(os.pardir, 'data/')

#   returns the directories of the data files
    dir_train_1, dir_train_2, dir_train_3, dir_test, dir_add_1, dir_add_2, dir_add_3 = get_file_paths()
    print('got list dirs')

#   Lists of the image paths
    list_abspath_img_train_1 = get_list_abspath_img(dir_train_1)
    train_1_labels = [1]*len(list_abspath_img_train_1)
    list_abspath_img_train_2 = get_list_abspath_img(dir_train_2)
    train_2_labels = [2]*len(list_abspath_img_train_2)
    list_abspath_img_train_3 = get_list_abspath_img(dir_train_3)
    train_3_labels = [3]*len(list_abspath_img_train_3)
    train_lists  = list_abspath_img_train_1 + list_abspath_img_train_2 + list_abspath_img_train_3
    train_labels  = train_1_labels + train_2_labels + train_3_labels  

#   Create/Save train df
    t_dict = {'paths': pd.Series(train_lists),
              'labels':pd.Series(train_labels)}
    train_df = pd.DataFrame(t_dict)
    save_dirs(os.path.join(dir,'train.csv'), train_df, 'df')
    print('saved train dirs')

#   Create/Save test df
    test_list = get_list_abspath_img(dir_test)
    test_df = pd.DataFrame({'paths' : pd.Series(test_list)})
    save_dirs(os.path.join(dir,'test.csv'), test_df, 'df')
    print('saved test dirs')

#   Create/Save addtionals df
    list_abspath_img_add_1   = get_list_abspath_img(dir_add_1)
    add_1_labels = [1]*len(list_abspath_img_add_1)
    list_abspath_img_add_2   = get_list_abspath_img(dir_add_2)
    add_2_labels = [2]*len(list_abspath_img_add_2)
    list_abspath_img_add_3   = get_list_abspath_img(dir_add_3)
    add_3_labels = [3]*len(list_abspath_img_add_3)
    add_lists = list_abspath_img_add_1   + list_abspath_img_add_2   + list_abspath_img_add_3
    add_labels = add_1_labels + add_2_labels + add_3_labels
    add_dict = {'paths': pd.Series(add_lists),
              'labels':pd.Series(add_labels)}
    add_df = pd.DataFrame(add_dict)
    save_dirs(os.path.join(dir,'additionals.csv'), add_df, 'df')
    print('saved additional dirs')


def main():
    save_img_dirs()

if __name__ == '__main__':
    main()
