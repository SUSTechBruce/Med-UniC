import numpy as np
import pandas as pd
from mgca.constants import *

np.random.seed(0)

import os.path as osp
import os
os.path.abspath('.')


def filtering_chexpert():
    df = pd.read_csv(CHEXPERT_ORIGINAL_TRAIN_CSV)
    total_paths = df['Path']
    
    t_count = 0
    
    filter_list = []
    
    for i, path in enumerate(total_paths):
        path = path.replace('CheXpert-v1.0/', '')
        new_path = os.path.join(CHEXPERT_DATA_DIR, path)
        
        df['Path'][i] = 'CheXpert-v1.0/' + path
        
        if os.path.exists(new_path):
            print('yes')
            
        else:
            
            t_count += 1
            print(path)
            filter_list.append('CheXpert-v1.0/' + path)
            
    print('Count the filter numbers: ', t_count)
            
    df = df[~df['Path'].isin(filter_list)]
    
    df = df.reset_index(drop=True)
    
    df.to_csv(CHEXPERT_fillter)

    return df

if __name__ == '__main__':
    filtering_chexpert()