# -*- coding: utf-8 -*-

import pandas as pd
import os
import uuid
import cv2
import albumentations as A
from sklearn.model_selection import train_test_split
from autogluon.multimodal import MultiModalPredictor
import numpy as np
import matplotlib.pyplot as plt
from gen_csv import natural_sort_key

# 数据下载目录
download_dir = './road_segmentation'
load = False

dataset_path = download_dir
train_data = pd.read_csv(f'{dataset_path}/train.csv', index_col=0)
test_data = pd.read_csv(f'{dataset_path}/test.csv', index_col=0)

image_col = 'image'
label_col = 'label'

def path_expander(path, base_folder):
    path_l = path.split(';')
    return ';'.join([os.path.abspath(os.path.join(base_folder, path)) for path in path_l])

for per_col in [image_col, label_col]:
    train_data[per_col] = train_data[per_col].apply(lambda ele: path_expander(ele, base_folder='./'))
test_data[image_col] = test_data[image_col].apply(lambda ele: path_expander(ele, base_folder='./'))

# 数据增强设置
augmentation_times = 20

transform = A.Compose([
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
    A.MotionBlur(blur_limit=5, p=0.3)
])

# 原始训练数据
train_dataset = os.path.join(dataset_path, 'train_original')
images_folder = os.path.join(train_dataset, 'images')
labels_folder = os.path.join(train_dataset, 'groundtruth')

image_files = sorted(
    [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')], key=natural_sort_key)
label_files = sorted(
    [os.path.join(labels_folder, f) for f in os.listdir(labels_folder) if f.endswith('.jpg') or f.endswith('.png')], key=natural_sort_key)

# 生成增强数据的函数
def augment_data(seed):
    np.random.seed(seed)
    cnt = len(image_files)
    aug_images = []
    aug_labels = []
    AUG_DATA_FOLDER_IMG = "./road_segmentation/aug/images"
    AUG_DATA_FOLDER_GT = "./road_segmentation/aug/groundtruth"

    if not os.path.exists(AUG_DATA_FOLDER_IMG):
        os.makedirs(AUG_DATA_FOLDER_IMG)
        print(f"Directory {AUG_DATA_FOLDER_IMG} created.")

    if not os.path.exists(AUG_DATA_FOLDER_GT):
        os.makedirs(AUG_DATA_FOLDER_GT)
        print(f"Directory {AUG_DATA_FOLDER_GT} created.")

    for i in range(augmentation_times):
        for j in range(len(image_files)):
            image = cv2.imread(image_files[j])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(label_files[j])
            transformed = transform(image=image, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']

            if len(transformed_mask.shape) == 3:
                transformed_mask = cv2.cvtColor(transformed_mask, cv2.COLOR_RGB2GRAY)

            image_path = f'./road_segmentation/aug/images/satimage_{cnt}.png'
            label_path = f'./road_segmentation/aug/groundtruth/satimage_{cnt}.png'
            cv2.imwrite(image_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(label_path, transformed_mask)

            aug_images.append(image_path)
            aug_labels.append(label_path)

            print(f'satimage_{cnt} augmentation saved')
            cnt += 1

    return pd.DataFrame({image_col: aug_images, label_col: aug_labels})

# 训练多个模型并保存 checkpoint
num_models = 10
checkpoints = []

for i in range(num_models):
    save_path = f"./tmp/{uuid.uuid4().hex}-automm_semantic_seg"
    
    predictor = MultiModalPredictor(
        problem_type="semantic_segmentation",
        label="label",
        hyperparameters={
            "model.sam.checkpoint_name": "facebook/sam-vit-huge",
        },
        path=save_path,
        presets="best_quality",
    )

    augmented_train_data = augment_data(seed=i)
    combined_train_data = pd.concat([train_data, augmented_train_data], ignore_index=True)
    _, val_data = train_test_split(combined_train_data, test_size=0.2, random_state=42)
    
    hyperparameters = {
        'optimization': {
            # 'max_epochs': 100,
            'max_epochs': 50,
        },
    }

    predictor.fit(
        train_data=combined_train_data,
        tuning_data=val_data,
        presets="best_quality",
        # time_limit=1000,
        # hyperparameters=hyperparameters,
    )
    
    predictor.save(save_path)
    checkpoints.append(save_path)

    del predictor

# 加载保存的 checkpoint 并进行集成
predictors = [MultiModalPredictor.load(path) for path in checkpoints]

# 创建集成预测函数
def ensemble_predict(predictors, data):
    preds = []
    for predictor in predictors:
        # pred = predictor.predict(data)  # 使用 predict 获取预测值
        pred = predictor.predict_proba(data, as_multiclass=False)  # 使用 predict 获取预测值
        if isinstance(pred, list):
            pred = np.array(pred)
        preds.append(pred)
    avg_preds = np.mean(preds, axis=0)  # 对每个预测值取平均
    predict = avg_preds > 0.5
    return predict.squeeze(axis=1)  # 返回平均值作为最终预测

# 进行预测
test_images = test_data[image_col].tolist()
ensemble_predictions = ensemble_predict(predictors, {'image': test_images})
print(ensemble_predictions.shape)

def save_predictions(predictions, test_images):
    for i in range(len(predictions)):
        binary_mask = np.array(predictions[i], dtype=np.uint8) * 255
        print(test_images[i].replace('images', 'groundtruth'))
        plt.imsave(test_images[i].replace('images', 'groundtruth'), binary_mask[0], cmap='gray')

save_predictions(ensemble_predictions, test_images)
