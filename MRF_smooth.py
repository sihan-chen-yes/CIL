import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_gaussian, create_pairwise_bilateral
import cv2
from gen_csv import natural_sort_key
import os

dataset_path = './road_segmentation'

# gen train dataset data frame
test_dataset = os.path.join(dataset_path, 'test')

images_folder = os.path.join(test_dataset, 'images')
labels_folder = os.path.join(test_dataset, 'groundtruth')

image_files = sorted(
    [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')],
    key=natural_sort_key)

label_files = sorted(
    [os.path.join(labels_folder, f) for f in os.listdir(labels_folder) if f.endswith('.jpg') or f.endswith('.png')],
    key=natural_sort_key)

assert len(image_files) == len(label_files)

SMOOTH_GT = "./road_segmentation/test/smooth_gt"

if not os.path.exists(SMOOTH_GT):
    os.makedirs(SMOOTH_GT)
    print(f"Directory {SMOOTH_GT} created.")

for i in range(len(image_files)):
    image = cv2.imread(image_files[i], cv2.COLOR_BGR2RGB)

    label = cv2.imread(label_files[i])
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    thresh = 128
    _, label = cv2.threshold(label, thresh, 1, cv2.THRESH_BINARY)
    label = label.flatten()
    # dcrf object
    # 2 classes
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)

    unary = unary_from_labels(label, 2, gt_prob=0.8, zero_unsure=False)
    d.setUnaryEnergy(unary)

    # bilateral energy
    # smoothing item
    pairwise_gaussian = create_pairwise_gaussian(sdims=(3, 3), shape=image.shape[:2])
    d.addPairwiseEnergy(pairwise_gaussian, compat=2)

    pairwise_bilateral = create_pairwise_bilateral(sdims=(5, 5), schan=(1, 1, 1), img=image, chdim=2)
    d.addPairwiseEnergy(pairwise_bilateral, compat=1)

    #inference times
    Q = d.inference(20)
    smoothed_label = np.argmax(Q, axis=0).reshape(image.shape[:2])

    file = label_files[i].replace('groundtruth', 'smooth_gt')
    smoothed_label = np.array(smoothed_label, dtype=np.uint8) * 255
    cv2.imwrite(file, cv2.cvtColor(smoothed_label, cv2.COLOR_RGB2BGR))
    print(f'{file} saved')
