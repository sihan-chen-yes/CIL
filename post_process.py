import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_gaussian, create_pairwise_bilateral
import cv2
from gen_csv import natural_sort_key
import os
import maxflow
from enum import Enum

class Method(Enum):
    CRF = 0
    GRAPH_CUT = 1

post_process_method = Method.GRAPH_CUT

def CRF(label, gt_prob=0.6, gaussian_sdims=3, gaussian_compat=2, bilateral_sdims=5, bilateral_compat=1, bilateral_schan=5, iterations=20):
    """
    - gt_prob: Ground truth probability for the unary term.
        A higher value indicates more confidence in the initial labels
        0~1
    - gauss_sdims: Standard deviations for the Gaussian pairwise term.
        Larger values lead to more extensive spatial smoothing.
        2~10
    - gauss_compat: Compatibility coefficient for the Gaussian pairwise term.
        A higher value increases the influence of the Gaussian smoothing term,
        leading to a smoother result.
        1~10
    - bilateral_sdims: Spatial standard deviations for the bilateral pairwise term.
        Similar to gauss_sdims, but also considers color information.
        Larger values lead to more spatial smoothing.
        5~10
    - bilateral_schan: Color standard deviations for the bilateral pairwise term.
        Determines the extent of smoothing in the color space.
        Larger values cause pixels with similar colors to be grouped together
        5~15
    - bilateral_compat: Compatibility coefficient for the bilateral pairwise term.
        A higher value increases the influence of the bilateral smoothing term,
        which smooths regions with similar colors.
        1~10
    - iterations: Number of iterations for CRF inference.
        More iterations usually result in more accurate results but also increase computation time
        10~50
    """
    label = label.flatten()
    # dcrf object
    # 2 classes
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)

    unary = unary_from_labels(label, 2, gt_prob=gt_prob, zero_unsure=False)
    d.setUnaryEnergy(unary)

    # bilateral energy
    # smoothing item
    pairwise_gaussian = create_pairwise_gaussian(sdims=(gaussian_sdims, gaussian_sdims), shape=image.shape[:2])
    d.addPairwiseEnergy(pairwise_gaussian, compat=gaussian_compat)

    pairwise_bilateral = create_pairwise_bilateral(sdims=(bilateral_sdims, bilateral_sdims),
                                                   schan=(bilateral_schan, bilateral_schan, bilateral_schan),
                                                   img=image, chdim=2)
    d.addPairwiseEnergy(pairwise_bilateral, compat=bilateral_compat)

    # inference times
    Q = d.inference(iterations)
    smoothed_label = np.argmax(Q, axis=0).reshape(image.shape[:2])

    return smoothed_label

def GRAPH_CUT(label, lambda_param=2, edge_weight=1.5):
    """
    lambda_param: penalty for change the label
    greater means bigger penalty for changing wrongly
    tend to follow original label
    edge_weight: bigger, smoother
    smaller, more details preserved
    """

    rows, cols = label.shape
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((rows, cols))

    # Define the structure for 4-connectivity
    structure = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.int32)

    # Adding edges for the 4-connectivity structure
    g.add_grid_edges(nodeids, structure=structure, weights=edge_weight)

    # Adding terminal edges
    for i in range(rows):
        for j in range(cols):
            if label[i, j] == 1:  # If the pixel is predicted as street
                g.add_tedge(nodeids[i, j], 0, lambda_param)
            else:  # If the pixel is predicted as non-street
                g.add_tedge(nodeids[i, j], lambda_param, 0)

    g.maxflow()

    smoothed_label = g.get_grid_segments(nodeids)

    return smoothed_label

if __name__ == '__main__':

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

        if post_process_method == Method.CRF:
            smoothed_label = CRF(label)
        else:
            smoothed_label = GRAPH_CUT(label)

        file = label_files[i].replace('groundtruth', 'smooth_gt')
        smoothed_label = np.array(smoothed_label, dtype=np.uint8) * 255
        cv2.imwrite(file, cv2.cvtColor(smoothed_label, cv2.COLOR_RGB2BGR))
        print(f'{file} saved')
