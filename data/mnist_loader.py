import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

def download_mnist_dataset(data_dir='data/demo_data/mnist'):
    """
    Download the MNIST dataset if it doesn't exist already.
    
    Args:
        data_dir: Path to save the dataset
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download training set
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # Download test set
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"Downloaded MNIST dataset - Training: {len(train_dataset)} samples, Test: {len(test_dataset)} samples")
    
    # Create directories for each class
    for digit in range(10):
        os.makedirs(os.path.join(data_dir, 'train', str(digit)), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'test', str(digit)), exist_ok=True)
    
    # Save training images (limit to a smaller subset for quantum computing)
    samples_per_class = 100  # Adjust as needed for your compute resources
    class_counts = [0] * 10
    
    print("Saving training images...")
    for idx, (img, label) in enumerate(train_dataset):
        digit = int(label)
        if class_counts[digit] < samples_per_class:
            # Convert tensor to PIL image
            img_np = img.squeeze().numpy()
            img_scaled = ((img_np * 0.3081) + 0.1307) * 255  # Unnormalize
            img_pil = Image.fromarray(img_scaled.astype(np.uint8), mode='L')
            
            # Save the image
            save_path = os.path.join(data_dir, 'train', str(digit), f'digit_{digit}_{class_counts[digit]}.png')
            img_pil.save(save_path)
            class_counts[digit] += 1
            
            if sum(class_counts) % 100 == 0:
                print(f"Saved {sum(class_counts)} training images")
                
        if all(count >= samples_per_class for count in class_counts):
            break
    
    # Save test images (smaller subset)
    test_samples_per_class = 20  # Adjust as needed
    class_counts = [0] * 10
    
    print("Saving test images...")
    for idx, (img, label) in enumerate(test_dataset):
        digit = int(label)
        if class_counts[digit] < test_samples_per_class:
            # Convert tensor to PIL image
            img_np = img.squeeze().numpy()
            img_scaled = ((img_np * 0.3081) + 0.1307) * 255  # Unnormalize
            img_pil = Image.fromarray(img_scaled.astype(np.uint8), mode='L')
            
            # Save the image
            save_path = os.path.join(data_dir, 'test', str(digit), f'digit_{digit}_{class_counts[digit]}.png')
            img_pil.save(save_path)
            class_counts[digit] += 1
            
            if sum(class_counts) % 50 == 0:
                print(f"Saved {sum(class_counts)} test images")
                
        if all(count >= test_samples_per_class for count in class_counts):
            break
            
    print(f"MNIST dataset preparation complete! Images saved to {data_dir}")
    return data_dir

def load_mnist_dataset(data_dir='data/demo_data/mnist', split='train', img_size=(28, 28)):
    """
    Load MNIST dataset for QNN Explorer
    
    Args:
        data_dir: Path to MNIST dataset directory
        split: 'train' or 'test'
        img_size: Image size for resizing
        
    Returns:
        images: Tensor of images [N, C, H, W]
        labels: Tensor of labels [N]
    """
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)  # Convert grayscale to 3 channels for CNN compatibility
    ])
    
    # Initialize lists for images and labels
    images = []
    labels = []
    
    print(f"Loading MNIST dataset from {data_dir}, split: {split}")
    
    # Process each digit class
    for digit in range(10):
        digit_dir = os.path.join(data_dir, split, str(digit))
        print(f"Checking digit directory: {digit_dir}")
        if os.path.exists(digit_dir):
            digit_files = [f for f in os.listdir(digit_dir) if f.endswith('.png')]
            print(f"Found {len(digit_files)} files for digit {digit}")
            
            for filename in digit_files:
                img_path = os.path.join(digit_dir, filename)
                try:
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img_tensor = transform(img)
                    images.append(img_tensor)
                    labels.append(digit)  # Digit is the class label
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    # Convert lists to tensors
    if images:
        images_tensor = torch.stack(images)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        print(f"Loaded {len(images)} images with shape {images_tensor.shape}")
        return images_tensor, labels_tensor
    else:
        print("No images found! Try downloading the dataset first.")
        return None, None

if __name__ == "__main__":
    # This will download the dataset when run directly
    download_mnist_dataset() 