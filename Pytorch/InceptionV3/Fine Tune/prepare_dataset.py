import os
import tarfile
import requests
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split

# Function to download and extract the dataset
def download_and_extract(url, dest_path):
    filename = url.split('/')[-1]
    tar_path = os.path.join(dest_path, filename)

    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(tar_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    # Extract the tar file
    with tarfile.open(tar_path, 'r') as tar_ref:
        tar_ref.extractall(dest_path)

# URLs for the dataset
images_url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
annotations_url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar'

# Destination path
dest_path = 'data/stanford_dogs'
os.makedirs(dest_path, exist_ok=True)

# Download and extract
download_and_extract(images_url, dest_path)
download_and_extract(annotations_url, dest_path)

# Define paths
images_path = os.path.join(dest_path, 'Images')
train_path = os.path.join(dest_path, 'train')
val_path = os.path.join(dest_path, 'val')

# Create train and val directories
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# Get all class folders
classes = [d for d in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, d))]

# Split each class into train and val
for cls in classes:
    cls_path = os.path.join(images_path, cls)
    images = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
    
    os.makedirs(os.path.join(train_path, cls), exist_ok=True)
    os.makedirs(os.path.join(val_path, cls), exist_ok=True)
    
    for img in train_images:
        shutil.move(os.path.join(cls_path, img), os.path.join(train_path, cls, img))
        
    for img in val_images:
        shutil.move(os.path.join(cls_path, img), os.path.join(val_path, cls, img))

# Clean up
shutil.rmtree(images_path)

print("Dataset prepared successfully.")
