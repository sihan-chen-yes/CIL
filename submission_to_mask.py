import numpy as np
import pandas as pd
from PIL import Image
import os

csv_file = "dummy_submission.csv"
df = pd.read_csv(csv_file)

images = {}

for index, row in df.iterrows():
    if index == 0:
        continue
    parts = row['id'].split('_')
    img_number = int(parts[0])
    col = int(parts[1])
    row_num = int(parts[2])
    label = int(row['prediction'])

    if img_number not in images:
        images[img_number] = np.zeros((400, 400), dtype=np.uint8)

    start_col = col
    start_row = row_num

    images[img_number][start_row:start_row + 16, start_col:start_col + 16] = label * 255

output_dir = './output_images'
os.makedirs(output_dir, exist_ok=True)

for img_number, image in images.items():
    output_filename = os.path.join(output_dir, f'image_{img_number}.png')
    img = Image.fromarray(image)
    img.save(output_filename)
    print(f'Image {img_number} saved to {output_filename}')
