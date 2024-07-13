import os
import pandas as pd
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

if __name__ == '__main__':
    # augmentation option
    AUG = True
    original_dataset_num = 144

    dataset_path = './road_segmentation'

    #gen train dataset data frame
    train_dataset = os.path.join(dataset_path, 'train')

    images_folder = os.path.join(train_dataset, 'images')
    labels_folder = os.path.join(train_dataset, 'groundtruth')

    image_files = sorted(
                [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')], key=natural_sort_key)
    label_files = sorted(
                [os.path.join(labels_folder, f) for f in os.listdir(labels_folder) if f.endswith('.jpg') or f.endswith('.png')], key=natural_sort_key)

    if not AUG:
        data = pd.DataFrame({
                'image': image_files[:original_dataset_num],
                'label': label_files[:original_dataset_num]
            })
    else:
        data = pd.DataFrame({
                'image': image_files,
                'label': label_files
            })
    train_csv = os.path.join(dataset_path, 'train.csv')
    data.to_csv(train_csv, index=True)
    print(f"The file '{train_csv}' is saved.")

    #gen test dataset data frame
    test_dataset = os.path.join(dataset_path, 'test')
    images_folder = os.path.join(test_dataset, 'images')
    # labels_folder = os.path.join(test_dataset, 'groundtruth')
    image_files = sorted(
                [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')])
    # label_files = [os.path.join(labels_folder, os.path.basename(f)) for f in image_files]
    data = pd.DataFrame({
            'image': image_files,
    })
    test_csv = os.path.join(dataset_path, 'test.csv')
    data.to_csv(test_csv, index=True)
    print(f"The file '{test_csv}' is saved.")


