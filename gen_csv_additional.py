import os
import pandas as pd
# load images and convert the convertion to grayscale

import numpy as np
from PIL import Image

dataset_path = './road_segmentation/collected'
# dataset_path = './road_segmentation/collected_new'

# name = '0_ZOOM_18'
# name = '1_ZOOM_18'

#gen train dataset data frame

def save_name(name):
    train_dataset = os.path.join(dataset_path, name)

    image_files = sorted(
                [os.path.join(train_dataset, f) for f in os.listdir(train_dataset) if f.endswith('.jpg') or f.endswith('.png') and 'label' not in f], key=lambda x: int(x.split('/')[-1].split('.')[0]))
    label_files = sorted(
                [os.path.join(train_dataset, f) for f in os.listdir(train_dataset) if f.endswith('.jpg') or f.endswith('.png') and 'label' in f], key=lambda x: int(x.split('/')[-1].split('_')[0]))
    # image_files = sorted(image_files)
    # label_files = sorted(label_files)
    data = pd.DataFrame({
            'image': image_files,
            'label': label_files
        })
    train_csv = os.path.join(dataset_path, f'{name}.csv')
    data.to_csv(train_csv, index=True)
    print(f"The file '{train_csv}' is saved.")



    for label_file in label_files:
        img = Image.open(label_file)
        img = img.convert('L')
        img.save(label_file)


all_dir = os.listdir(dataset_path)
for name in all_dir:
    if os.path.isdir(os.path.join(dataset_path, name)):
        save_name(name)




