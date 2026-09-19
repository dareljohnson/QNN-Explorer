import os
import sys
import torch
import unittest
import tempfile
from PIL import Image
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mnist_loader import load_mnist_dataset

class TestMNISTChannels(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Create necessary subdirectories
        os.makedirs(os.path.join(self.temp_dir, 'train', '0'), exist_ok=True)
        
        # Create a dummy grayscale image
        img_size = (28, 28)
        blank_img = Image.new('L', img_size, color=255)
        self.test_img_path = os.path.join(self.temp_dir, 'train', '0', 'digit_0_0.png')
        blank_img.save(self.test_img_path)
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_channel_conversion(self):
        """Test that grayscale MNIST images are converted to 3 channels."""
        # Load the dataset with our temporary image
        images, labels = load_mnist_dataset(self.temp_dir, 'train')
        
        # Check that we have images and labels
        self.assertIsNotNone(images, "Images should not be None")
        self.assertIsNotNone(labels, "Labels should not be None")
        
        # Check number of channels (should be 3 after conversion)
        self.assertEqual(images.shape[1], 3, 
                         f"Expected 3 channels but got {images.shape[1]}")
        
        # Verify that all three channels have the same content (repeated)
        first_img = images[0]
        channel0 = first_img[0]
        channel1 = first_img[1]
        channel2 = first_img[2]
        
        # Check that all channels are identical (our Lambda function repeats the same channel)
        self.assertTrue(torch.allclose(channel0, channel1), 
                        "Channel 0 and 1 should be identical")
        self.assertTrue(torch.allclose(channel1, channel2), 
                        "Channel 1 and 2 should be identical")
        
        print(f"✅ MNIST images correctly converted from 1 to 3 channels: {images.shape}")

if __name__ == "__main__":
    unittest.main() 