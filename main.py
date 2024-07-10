# -*- coding: utf-8 -*-

download_dir = './road_segmentation'

import pandas as pd
import os
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


# print(train_data[image_col].iloc[0])
# print(test_data[image_col].iloc[0])

"""Each Pandas DataFrame contains two columns: one for image paths and the other for corresponding groundtruth masks. Let's take a closer look at the train data DataFrame."""

# train_data.head()


"""We can also visualize one image and its groundtruth mask."""

from autogluon.multimodal.utils import SemanticSegmentationVisualizer
visualizer = SemanticSegmentationVisualizer()
# visualizer.plot_image(train_data.iloc[0]['image'])
#
# visualizer.plot_image(train_data.iloc[0]['label'])

"""## Zero Shot Evaluation

Now, let's see how well the pretrained SAM can segment the images. For this demonstration, we'll use the base SAM model.
"""

from autogluon.multimodal import MultiModalPredictor
# predictor_zero_shot = MultiModalPredictor(
#     problem_type="semantic_segmentation",
#     label=label_col,
#      hyperparameters={
#             "model.sam.checkpoint_name": "facebook/sam-vit-base",
#         },
#     num_classes=1, # forground-background segmentation
# )

"""After initializing the predictor, you can perform inference directly."""
#
# pred_zero_shot = predictor_zero_shot.predict({'image': [test_data.iloc[0]['image']]})
#
# visualizer.plot_mask(pred_zero_shot)

"""It's worth noting that SAM without prompts outputs a rough leaf mask instead of disease masks due to its lack of context about the domain task. While SAM can perform better with proper click prompts, it might not be an ideal end-to-end solution for some applications that require a standalone model for deployment.

You can also conduct a zero-shot evaluation on the test data.
"""

# scores = predictor_zero_shot.evaluate(train_data, metrics=["iou"])
# print(scores)

"""As expected, the test score of the zero-shot SAM is relatively low. Next, let's explore how to fine-tune SAM for enhanced performance.

## Finetune SAM

Initialize a new predictor and fit it with the train and validation data.
"""

from autogluon.multimodal import MultiModalPredictor
import uuid
save_path = f"./tmp/{uuid.uuid4().hex}-automm_semantic_seg"
predictor = MultiModalPredictor(
    problem_type="semantic_segmentation",
    label="label",
     hyperparameters={
            "model.sam.checkpoint_name": "facebook/sam-vit-base",
        },
    # num_classes=1,
    path=save_path,
)
predictor.fit(
    train_data=train_data,
    # tuning_data=val_data,
    time_limit=3600, # seconds
)

"""Under the hood, we use [LoRA](https://arxiv.org/abs/2106.09685) for efficient fine-tuning. Note that, without hyperparameter customization, the huge SAM serves as the default model, which requires efficient fine-tuning in many cases.

After fine-tuning, evaluate SAM on the test data.
"""

# scores = predictor.evaluate(train_data, metrics=["iou"])
# print(scores)

"""Thanks to the fine-tuning process, the test score has significantly improved.

To visualize the impact, let's examine the predicted mask after fine-tuning.
"""
test_images = test_data['image'].tolist()
predictions = predictor.predict({'image': test_images})
# save images
import numpy as np
import matplotlib.pyplot as plt
dir_path = './road_segmentation/test/groundtruth'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print(f"Directory {dir_path} created.")

for i in range(len(predictions)):
    binary_mask = np.array(predictions[i], dtype=np.uint8) * 255
    print(test_images[i].replace('images', 'groundtruth'))
    plt.imsave(test_images[i].replace('images', 'groundtruth'), binary_mask[0], cmap='gray')

# visualizer.plot_mask(pred)

"""As evident from the results, the predicted mask is now much closer to the groundtruth. This demonstrates the effectiveness of using AutoMM to fine-tune SAM for domain-specific applications, enhancing its performance in tasks like leaf disease segmentation.

## Save and Load

The trained predictor is automatically saved at the end of `fit()`, and you can easily reload it.

```{warning}

`MultiModalPredictor.load()` uses `pickle` module implicitly, which is known to be insecure. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling. Never load data that could have come from an untrusted source, or that could have been tampered with. **Only load data you trust.**

```
"""

# loaded_predictor = MultiModalPredictor.load(save_path)
# scores = loaded_predictor.evaluate(train_data, metrics=["iou"])
# print(scores)

"""We can see the evaluation score is still the same as above, which means same model!

## Other Examples

You may go to [AutoMM Examples](https://github.com/autogluon/autogluon/tree/master/examples/automm) to explore other examples about AutoMM.

## Customization
To learn how to customize AutoMM, please refer to [Customize AutoMM](../advanced_topics/customization.ipynb).
"""