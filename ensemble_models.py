# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from autogluon.multimodal import MultiModalPredictor

# 数据下载目录
download_dir = './road_segmentation'

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

# 手动读取tmp目录下的所有checkpoint
checkpoint_paths = [os.path.join("./tmp", f) for f in os.listdir("./tmp") if f.endswith("-automm_semantic_seg")]
checkpoint_paths += [os.path.join("/mnt/sda/tmp", f) for f in os.listdir("./tmp") if f.endswith("-automm_semantic_seg-ensemble")]
# predictors = [MultiModalPredictor.load(path) for path in checkpoint_paths]
predictors  = []
for path in checkpoint_paths:
    try:
        predictor = MultiModalPredictor.load(path)
        predictors.append(predictor)
    except Exception as e:
        print(f"Failed to load predictor from {path}: {e}")
        continue

# 创建集成预测函数
def ensemble_predict(predictors, data):
    preds = []
    for predictor in predictors:
        pred = predictor.predict_proba(data, as_multiclass=False)
        if isinstance(pred, list):
            pred = np.array(pred)
        preds.append(pred)
    avg_preds = np.mean(preds, axis=0)
    predict = avg_preds > 0.5
    return predict.squeeze(axis=1)

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
