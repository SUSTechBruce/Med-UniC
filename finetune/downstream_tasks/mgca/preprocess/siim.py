import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from mgca.constants import *

# The missing df200 counts is:  84
# The missing training counts is:  20732
# Number of train samples: 217433
# Number of valid samples: 5000
# Number of chexpert5x200 samples: 1000


np.random.seed(13)

def preprocess_pneumothorax_data(test_fac=0.15):
    try:
        df = pd.read_csv(PNEUMOTHORAX_ORIGINAL_TRAIN_CSV)
    except:
        raise Exception(
            "Please make sure the the SIIM Pneumothorax dataset is \
            stored at {PNEUMOTHORAX_DATA_DIR}"
        )

    # get image paths
    os.listdir(PNEUMOTHORAX_IMG_DIR)
    img_paths = {}
    for subdir, dirs, files in tqdm(os.walk(PNEUMOTHORAX_IMG_DIR)):
        for f in files:
            if "dcm" in f:
                # remove dcm
                file_id = f[:-4]
                # img_paths[file_id] = os.path.join(subdir[105:], f)
                img_paths[file_id] = os.path.join(subdir, f)
    # no encoded pixels mean healthy
    df["Label"] = df.apply(
        lambda x: 0.0 if x[" EncodedPixels"] == " -1" else 1.0, axis=1
    )
    df["Path"] = df["ImageId"].apply(lambda x: img_paths[x])

    # split data
    train_df, test_val_df = train_test_split(
        df, test_size=test_fac * 2, random_state=42)
    test_df, valid_df = train_test_split(
        test_val_df, test_size=0.5, random_state=42)

    print(f"Number of train samples: {len(train_df)}")
    print(train_df["Label"].value_counts())
    print(f"Number of valid samples: {len(valid_df)}")
    print(valid_df["Label"].value_counts())
    print(f"Number of test samples: {len(test_df)}")
    print(test_df["Label"].value_counts())

    train_df.to_csv(PNEUMOTHORAX_TRAIN_CSV, index=False)
    valid_df.to_csv(PNEUMOTHORAX_VALID_CSV, index=False)
    test_df.to_csv(PNEUMOTHORAX_TEST_CSV, index=False)


if __name__ == "__main__":
    preprocess_pneumothorax_data()
