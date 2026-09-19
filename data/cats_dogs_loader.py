from torchvision import transforms
from PIL import Image
import os
import torch

def load_cats_dogs_dataset(data_dir='data/demo_data/cats_dogs', split='train', img_size=(224, 224)):
    """
    Load cats vs dogs dataset for QNN Explorer
    
    Args:
        data_dir: Path to cats_dogs dataset directory
        split: 'train' or 'test'
        img_size: Image size for resizing
        
    Returns:
        images: Tensor of images [N, C, H, W]
        labels: Tensor of labels [N]
    """
    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize lists for images and labels
    images = []
    labels = []
    
    # Process cat images (label 0)
    cat_dir = os.path.join(data_dir, split, 'cat')
    if os.path.exists(cat_dir):
        print(f"Loading cat images from {cat_dir}")
        for filename in os.listdir(cat_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(cat_dir, filename)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)
                    images.append(img_tensor)
                    labels.append(0)  # Cat label
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    else:
        print(f"Warning: cat directory not found at {cat_dir}")
    
    # Process dog images (label 1)
    dog_dir = os.path.join(data_dir, split, 'dog')
    if os.path.exists(dog_dir):
        print(f"Loading dog images from {dog_dir}")
        for filename in os.listdir(dog_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(dog_dir, filename)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)
                    images.append(img_tensor)
                    labels.append(1)  # Dog label
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    else:
        print(f"Warning: dog directory not found at {dog_dir}")
    
    # Convert lists to tensors
    if images:
        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        print(f"Loaded {len(images)} images with shape {images_tensor.shape}")
        return images_tensor, labels_tensor
    else:
        print("No images found! Make sure to run download_cats_dogs.py first.")
        return None, None 