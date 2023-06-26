import pandas as pd
import os.path as osp
import os 

import subprocess
import numpy as np
import os
import pandas as pd
from PIL import Image
import h5py
import matplotlib.pyplot as plt
from typing import List

import torch
from torch.utils import data
from tqdm.notebook import tqdm
import torch.nn as nn
from torchvision.transforms import Compose, Normalize

import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, auc, roc_auc_score, roc_curve, classification_report
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import average_precision_score

import sys
# sys.path.append('../..')
# sys.path.append('../data-process')
# sys.path.append('data/padchest')

from data_preprocess import * 



def preprocess_data(data_root):
    labels_path = os.path.join(data_root, 
                            'PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv')
    labels = pd.read_csv(labels_path)
    # get filepaths of 2.zip images
    text_file_path = os.path.join(data_root, '2.zip.unzip-l.txt')
    image_paths = extract_filenames(text_file_path)
    labels_2_df = labels[labels['ImageID'].isin(image_paths)]
    unique_labels = get_unique_labels(labels_2_df)
    # multi hot encoding for labels
    df_lab = create_multi_hot_labels(labels_2_df, unique_labels)
    
    loc_2_df = labels[labels['ImageID'].isin(image_paths)]
    loc_col_2 = loc_2_df.loc[:, "Labels"] 
    # multihot encoding for localizations
    unique_loc = get_unique_labels(loc_2_df, column="Labels")
    df_loc = create_multi_hot_labels(loc_2_df, unique_loc, column="Labels")
    directory = 'data/padchest/images/'
    cxr_paths = get_paths(directory)
    write_h5(cxr_paths)
    unique_labels = np.load('unique_labels.npy')
    return unique_labels[0:1]


def extract_filenames(txt_path): 
    """
    Given a filepath to a txt file with image file names, 
    extract a list of filenames for this zip. 
    
    Assume that the txt file has two unnecessary lines at 
    both the top and the bottom of the file.
    """
    df = pd.read_csv(txt_path)
    df_list = df.values.tolist()
    df_list = df_list[2:-2]
    
    images_list = []
    for file in df_list: 
        parsed_filename = file[0].split()[-1]
        images_list.append(parsed_filename)
    return images_list

# get paths of all possible labels 
def get_unique_labels(labels_df, column='Labels'): 
    """
    Given labels_df, return a list containing all unique labels
    present in this dataset.
    """
    
    unique_labels = set()
    # iterate through all rows in the dataframe
    for index, row in labels_df.iterrows():
        labels = row[column]
        try: 
            # convert labels str to array
            labels_arr = labels.strip('][').split(', ')
            for label in labels_arr: 
                # process string
                processed_label = label.split("'")[1].strip()
                processed_label = processed_label.lower()
                unique_labels.add(processed_label)
        except: 
            continue
    print(unique_labels)
    return list(unique_labels)

def create_multi_hot_labels(labels_df, unique_labels_list, column='Labels'):
    """
    Args: 
        * labels_df: original df where labels are an arr
        * labels_list: list of all possible labels in respective order
        
    Given all entries and it's corresponding labels, create a one(multi)-hot vector
    where a 1 represents the presence of that disease.
    
    Returns a Pandas dataframe mapping filename to it's multi-hot representation. Each of the diseases
    are columns.
    """
    
    # todo: check how the labels are represented for CheXpert
    # create a pandas datafraame with columns as unique labels, start with list of dicts
    dict_list = []
    
    # iterate through all rows in the dataframe
    for index, row in labels_df.iterrows():
        labels = row[column]
        try: 
            # convert labels str to array
            labels_arr = labels.strip('][').split(', ')
#             print(labels_arr, len(labels_arr))
            
            count_dict = dict() # map label name to count
            count_dict['ImageID'] = row['ImageID']
             # init count dict with 0s
            for unq_label in unique_labels_list: 
                    count_dict[unq_label] = 0
                    
            if len(labels_arr) > 0 and labels_arr[0] != '': 
                for label in labels_arr: 
                    # process string
                    processed_label = label.split("'")[1].strip()
                    processed_label = processed_label.lower()
                    count_dict[processed_label] = 1
            
            dict_list.append(count_dict)
        except: 
            print("error when creating labels for this img.")
            continue
        
    multi_hot_labels_df = pd.DataFrame(dict_list, columns=(['ImageID'] + unique_labels_list))
    return multi_hot_labels_df

# convert folder of images to h5 file
def get_paths(directory): 
    """
    Given a directory, this function outputs 
    all the image paths in that directory as a 
    list.
    """
    paths_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            paths_list.append(os.path.join(directory, filename))
        else:
            continue
    return paths_list

def img_to_h5(
    cxr_paths: List[str], 
    out_filepath: str, 
    resolution: int = 320, 
) -> List[str]: 
    """
    Converts a set of images into a single `.h5` file. 
    
    Args: 
        cxr_paths: List of paths to images as `.png`
        out_filepath: Path to store h5 file
        resolution: image resolution
        
    Returns a list of cxr_paths that were successfully stored in the
    `.h5` file. 
    """
    dset_size = len(cxr_paths)
    proper_cxr_paths = []
    with h5py.File(out_filepath,'w') as h5f:
        img_dset = h5f.create_dataset('cxr', shape=(dset_size, resolution, resolution)) 

        ctr = 0
        for idx, path in enumerate(tqdm(cxr_paths)):
            try: 
                # read image using cv2
                img = cv2.imread(path)
                # convert to PIL Image object
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                # preprocess
                img = preprocess(img_pil, desired_size=resolution)     
                img_dset[ctr] = img
                ctr += 1
                proper_cxr_paths.append(path)
            except: 
                print(f"Image {ctr} failed loading...")
                continue
        print(h5f)
        
    return proper_cxr_paths

def write_h5(cxr_paths, resolution: int = 320):
#     out_filepath = 'data/padchest/images/2_cxr_dset_sample.h5'
#     dset_size = len(cxr_paths)

#     proper_cxr_paths = []
#     with h5py.File(out_filepath,'w') as h5f:
#         img_dset = h5f.create_dataset('cxr', shape=(2978, resolution, resolution)) # todo: replace magic number with actual number
#     #         print('Dataset initialized.')

#         ctr = 0
#         for idx, path in enumerate(tqdm(cxr_paths)):
#             try: 
#                 # read image using cv2
#                 img = cv2.imread(path)
#                 # convert to PIL Image object
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 img_pil = Image.fromarray(img)
#                 # preprocess
#                 img = preprocess(img_pil, desired_size=resolution)     
#                 plt.imshow(img)
#                 img_dset[ctr] = img
#                 ctr += 1
#                 proper_cxr_paths.append(path)
#             except: 
#                 print("failed!")
#                 continue
#         print(h5f)
#     np.save("proper_cxr_paths.npy", np.array(proper_cxr_paths))
#     out_filepath = 'data/padchest/images/2_cxr.h5'
#     img_to_hdf5(cxr_paths, out_filepath, resolution=320)
    df_labels_new = order_labels(df_lab, proper_cxr_paths)
    labels_path = 'data/padchest/2_cxr_labels.csv'
    df_labels_new.to_csv(labels_path)
        
def order_labels(df, cxr_paths): 
    """
    Fixes multi-hot labels to be in order of cxr_paths
    """
    df_new = pd.DataFrame(columns=df.columns)
    for path in cxr_paths: 
        imageId = path.split('/')[-1]
        row = df.loc[df['ImageID'] == imageId]
        df_new = df_new.append(row)
    return df_new

def preprocess_data_for_pdc_zeroshot(data_root):
    
    labels = pd.read_csv(data_root)
    # get filepaths of 2.zip images
#     text_file_path = os.path.join(data_root, '2.zip.unzip-l.txt')
#     image_paths = extract_filenames(text_file_path)
    labels_2_df = labels
    unique_labels = get_unique_labels(labels_2_df)
    # multi hot encoding for labels
    df_lab = create_multi_hot_labels(labels_2_df, unique_labels)
    
    print(df_lab)
    
#     loc_2_df = labels
#     loc_col_2 = loc_2_df.loc[:, "Labels"] 
#     # multihot encoding for localizations
#     unique_loc = get_unique_labels(loc_2_df, column="Labels")
#     df_loc = create_multi_hot_labels(loc_2_df, unique_loc, column="Labels")
#     directory = 'data/padchest/images/'
#     cxr_paths = get_paths(directory)
#     write_h5(cxr_paths)
#     unique_labels = np.load('unique_labels.npy')

    return df_lab


if __name__ == '__main__':
    pdc_path = osp.join(os.path.abspath('.'), "PDC_cleaned.csv")
    df_lab = preprocess_data_for_pdc_zeroshot(pdc_path)
    save_path = osp.join(os.path.abspath('.'), "zshit.csv")
    df_lab.to_csv(save_path, index=False)
