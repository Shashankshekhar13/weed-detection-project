import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = "AIML/dataset/images"

# Class distribution
class_names = os.listdir(data_dir)
print("Classes:", class_names)

# Count Images Per Class
class_counts = {class_name: len(os.listdir(os.path.join(data_dir, class_name))) for class_name in class_names}
print("Image Count Per Class:\n", class_counts)

# Plot Class Distribution
plt.bar(class_counts.keys(), class_counts.values())
plt.xticks(rotation=45)
plt.title("Class Distribution")
plt.show()