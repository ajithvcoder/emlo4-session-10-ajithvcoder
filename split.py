import os
import shutil
import random

# Define paths
base_dir = './data/PetImages'
train_dir = './data/train'
test_dir = './data/test'
categories = ['Cat', 'Dog']
split_ratio = 0.9  # 90% for training, 10% for testing

# Create train and test directories
for category in categories:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

# Split the images
for category in categories:
    category_path = os.path.join(base_dir, category)
    images = os.listdir(category_path)
    
    # Shuffle images
    random.shuffle(images)
    
    # Split into train and test sets
    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Move the images to the respective directories
    for img in train_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(train_dir, category, img))
    for img in test_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(test_dir, category, img))

print("Images have been split into train and test directories.")
