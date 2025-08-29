import os
import pandas as pd
import shutil

# Set paths relative to your project directory (/Users/sandhyasinha/Desktop/AIML)
source_dir = "images"              # Unsoroted images folder is in /Users/sandhyasinha/Desktop/AIML/images
csv_path = "labels.csv"         # CSV file in the project root (adjust if needed)
dest_dir = "AIML/dataset/images"   # Destination folder for sorted images

# Create the destination directory if it doesn't exist
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
    print(f"Created destination directory: {dest_dir}")

# Read the CSV file into a DataFrame and clean column names
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
print("CSV Columns:", df.columns.tolist())
print(df.head())

# For debugging: List files in source_dir
print("Files in source directory:", os.listdir(source_dir))

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    filename = os.path.basename(row["Filename"])
    class_label = row["Species"].strip()

    # Create the destination subfolder for this class if it doesn't exist
    class_dir = os.path.join(dest_dir, class_label)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
        print(f"Created directory: {class_dir}")

    # Build source and destination file paths
    src_file = os.path.join(source_dir, filename)
    dest_file = os.path.join(class_dir, filename)

    if os.path.exists(src_file):
        shutil.copy(src_file, dest_file)
        print(f"Copied {filename} to {class_label}")
    else:
        print(f"WARNING: {src_file} not found.")

print("Image sorting completed!")