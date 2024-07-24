# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import os

from autogluon.multimodal import MultiModalPredictor
import uuid
from sklearn.model_selection import train_test_split

def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

#directory to put dataset
download_dir = './road_segmentation'
load = False
# load ckpt
if load:
    id = "168a09ff805a4fc4b0e559b675e8832a"
    save_path = f"./tmp/{id}-automm_semantic_seg"
else:
    # not load ckpt
    save_path = f"./tmp/{uuid.uuid4().hex}-automm_semantic_seg"

dataset_path = download_dir
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)
image_col = 'image'
label_col = 'label'
additional_data = False

# directory to save test images
dir_path = './road_segmentation/test/groundtruth'

if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"Directory {dir_path} created.")

for per_col in [image_col, label_col]:
    train_data[per_col] = train_data[per_col].apply(lambda ele: path_expander(ele, base_folder='./'))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder='./'))

# with addtional training data
# print(train_data)
# print(train_data[image_col].iloc[0])
if additional_data:
    all_add_train_data = []

    all_add_dir = os.listdir(f'{dataset_path}/collected_new')
    all_add_csv = [i for i in all_add_dir if '.csv' in i]


    for name in all_add_csv:
        add_train_data = pd.read_csv(f'{dataset_path}/collected_new/{name}', index_col=0)
        for per_col in [image_col, label_col]:
            add_train_data[per_col] = add_train_data[per_col].apply(lambda ele: path_expander(ele, base_folder='./'))

        all_add_train_data.append(add_train_data)

    # merge all additional training data
    all_add_train_data = pd.concat(all_add_train_data, ignore_index=False)

    train_data = pd.concat([train_data, all_add_train_data], ignore_index=False)

if not load:
    predictor = MultiModalPredictor(
        problem_type="semantic_segmentation",
        label="label",
         hyperparameters={
                # "model.sam.checkpoint_name": "facebook/sam-vit-base",
                # "model.sam.checkpoint_name": "facebook/sam-vit-large",
                "model.sam.checkpoint_name": "facebook/sam-vit-huge",
            },
        # num_classes=1,
        path=save_path,
        presets="best_quality",
    )

    # self ensemble
    hyperparameters = {
        "optimization.top_k": 3,
        "optimization.top_k_average_method": "greedy_soup"
    }

    # hold out a part of training set as validation set
    _, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    # 设置训练配置，包括最大epoch数
    hyperparameters = {
        'optimization': {
            'max_epochs': 100,
        },
    }
    predictor.fit(
        train_data=train_data,
        tuning_data=val_data,
        # time_limit=3600, # seconds
        # HPO
        presets="best_quality",
        hyperparameters=hyperparameters,
        # hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
    )
else:
    predictor = MultiModalPredictor.load(save_path)
    print("loaded predictor")

test_images = test_data['image'].tolist()
predictions = predictor.predict({'image': test_images})


# save images
for i in range(len(predictions)):
    binary_mask = np.array(predictions[i], dtype=np.uint8) * 255
    print(test_images[i].replace('images', 'groundtruth'))
    plt.imsave(test_images[i].replace('images', 'groundtruth'), binary_mask[0], cmap='gray')

