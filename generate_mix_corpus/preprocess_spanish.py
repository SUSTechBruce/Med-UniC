import pandas as pd
import os.path as osp
import os 

if __name__ =='__main__':
    spanish_csv = osp.join(os.path.abspath('.'), "multiling_corpus/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv")
    report =  pd.read_csv(spanish_csv)['Report']
    report = list(report)
    print(report)
    print(len(report))
    
    