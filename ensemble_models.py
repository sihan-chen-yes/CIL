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
# checkpoint_paths = [os.path.join("./tmp", f) for f in os.listdir("./tmp") if f.endswith("-automm_semantic_seg")]
# checkpoint_paths += [os.path.join("/mnt/sda/tmp", f) for f in os.listdir("./tmp") if f.endswith("-automm_semantic_seg-ensemble")]
# checkpoint_paths += [os.path.join("/mnt/sda/tmp", f) for f in os.listdir("/mnt/sda/tmp") if f.endswith("-automm_semantic_seg")]
checkpoints_path = ['76c146382cec4716992fa7f92674cc79-automm_semantic_seg', 'c2a118e0d79b4385b5f78aa7e24d2eeb-automm_semantic_seg', 'a00a0689276e4a0680c4f790882d03ba-automm_semantic_seg']
checkpoints_path += ['0cde552fd94846a287426d85e27eea00-automm_semantic_seg', 'f69f86f88f724ab1b1d7a4235bf59e41-automm_semantic_seg', '835451512daa4375b08fae53f3e75452-automm_semantic_seg']
checkpoints_path += ['2c38a87002414bc5bbe30b726c647cc9-automm_semantic_seg', '93d1edf102394dc1a4d20fc45e41e3ba-automm_semantic_seg', 'a8d850b025d14ba0b9cb88b44ae356ed-automm_semantic_seg']
checkpoint_paths = [os.path.join("/mnt/sda/tmp", f) for f in checkpoints_path]
# predictors = [MultiModalPredictor.load(path) for path in checkpoint_paths]
print(f"Found {len(checkpoint_paths)} checkpoints")
predictors  = []
for path in checkpoint_paths:
    try:
        predictor = MultiModalPredictor.load(path)
        predictors.append(predictor)
    except Exception as e:
        print(f"Failed to load predictor from {path}: {e}")
        continue

print(f"Loaded {len(predictors)} predictors")

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
