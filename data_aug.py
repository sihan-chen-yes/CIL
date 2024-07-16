import albumentations as A
import cv2
import os
from gen_csv import natural_sort_key

# data augmentation options
augmentation_times = 3

transform = A.Compose([
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    # A.RandomResizedCrop(height=400, width=400, scale=(0.08, 1.0), ratio=(0.75, 1.33), p=0.5),
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
    A.MotionBlur(blur_limit=5, p=0.3)
])

# original train data
dataset_path = './road_segmentation'

#gen train dataset data frame
train_dataset = os.path.join(dataset_path, 'train_original')

images_folder = os.path.join(train_dataset, 'images')
labels_folder = os.path.join(train_dataset, 'groundtruth')

image_files = sorted(
            [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')], key=natural_sort_key)
label_files = sorted(
            [os.path.join(labels_folder, f) for f in os.listdir(labels_folder) if f.endswith('.jpg') or f.endswith('.png')], key=natural_sort_key)


num = len(image_files)
cnt = num
AUG_DATA_FOLDER_IMG = "./road_segmentation/aug/images"
AUG_DATA_FOLDER_GT = "./road_segmentation/aug/groundtruth"

if not os.path.exists(AUG_DATA_FOLDER_IMG):
    os.makedirs(AUG_DATA_FOLDER_IMG)
    print(f"Directory {AUG_DATA_FOLDER_IMG} created.")

if not os.path.exists(AUG_DATA_FOLDER_GT):
    os.makedirs(AUG_DATA_FOLDER_GT)
    print(f"Directory {AUG_DATA_FOLDER_GT} created.")

for i in range(augmentation_times):
    for j in range(num):
        image = cv2.imread(image_files[j])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(label_files[j])
        transformed = transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']

        if len(transformed_mask.shape) == 3:
            transformed_mask = cv2.cvtColor(transformed_mask, cv2.COLOR_RGB2GRAY)

        cv2.imwrite(f'./road_segmentation/aug/images/satimage_{cnt}.png', cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'./road_segmentation/aug/groundtruth/satimage_{cnt}.png', transformed_mask)

        print(f'satimage_{cnt} augmentation saved')
        cnt += 1