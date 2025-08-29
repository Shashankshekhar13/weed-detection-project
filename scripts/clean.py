# scripts/2_dataset_cleaning.py
import os
import shutil
import random
from collections import defaultdict
from tqdm import tqdm

input_dir = "AIML/dataset/images"
output_dir = "AIML/dataset_cleaned"
max_images_per_class = 1200  # Based on your graph, this is safe

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Organize images by class
class_to_images = defaultdict(list)
for class_name in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_name)
    if os.path.isdir(class_path):
        for img in os.listdir(class_path):
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                class_to_images[class_name].append(os.path.join(class_path, img))

# Downsample and copy
for class_name, images in tqdm(class_to_images.items()):
    output_class_path = os.path.join(output_dir, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    if len(images) > max_images_per_class:
        images = random.sample(images, max_images_per_class)

    for img_path in images:
        shutil.copy(img_path, output_class_path)

print("✅ Dataset cleaned and saved to:", output_dir)