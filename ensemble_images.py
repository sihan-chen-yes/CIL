import os
import numpy as np
from PIL import Image

def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            images[filename] = np.array(img)
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")
    return images

def weighted_average(images, weights):
    if not images:
        return None
    if len(images) != len(weights):
        raise ValueError("The number of images must match the number of weights.")
    
    # Initialize the weighted sum array
    weighted_sum = np.zeros_like(images[0], dtype=np.float64)
    
    # Sum up the weighted images
    for img, weight in zip(images, weights):
        weighted_sum += img * weight
    
    # Normalize by the sum of weights
    weighted_avg = np.clip(weighted_sum / sum(weights), 0, 255).astype(np.uint8)
    # weighted_avg = np.clip(weighted_sum, 0, 255).astype(np.uint8)
    
    return Image.fromarray(weighted_avg)

def main(folders, weights, output_path):
    all_images = {}
    
    for folder in folders:
        images = load_images_from_folder(folder)
        for filename, img in images.items():
            if filename not in all_images:
                all_images[filename] = []
            all_images[filename].append(img)
    
    if not all_images:
        print("No images found in the provided folders.")
        return

    for filename, images in all_images.items():
        if len(images) != len(folders):
            print(f"Skipping {filename} as it does not have corresponding images in all folders.")
            continue
        
        try:
            ensemble_image = weighted_average(images, weights)
            ensemble_image.save(os.path.join(output_path, filename))
            print(f"Ensemble image saved to {os.path.join(output_path, filename)}")
        except Exception as e:
            print(f"Error in processing image {filename}: {e}")

if __name__ == "__main__":
    folders = ["/home/yiming/Documents/class/CIL_project/CIL/road_segmentation/test/groundtruth", 
               "/home/yiming/Documents/class/CIL_project/CIL/road_segmentation/test/groundtruth_0715_vit_large", 
               "/home/yiming/Documents/class/CIL_project/CIL/road_segmentation/test/smooth_gt_0717_score92"]  # Replace with your folder paths
    weights = [0.7, 0.5, 0.3]  # Replace with your weights
    output_path = "/home/yiming/Documents/class/CIL_project/CIL/road_segmentation/ensemble_results/groundtruth"  # Replace with your output path
    os.makedirs(output_path, exist_ok=True)
    main(folders, weights, output_path)
