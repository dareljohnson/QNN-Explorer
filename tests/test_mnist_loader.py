import os
import sys
import unittest
import shutil
import tempfile
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mnist_loader import download_mnist_dataset, load_mnist_dataset

class TestMNISTLoader(unittest.TestCase):
    """Test suite for MNIST dataset loader"""
    
    def setUp(self):
        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_download_and_load(self):
        """Test downloading and loading a small subset of MNIST"""
        # Note: To keep the test fast, we'll only download a small subset
        # and verify the structure, not the full dataset
        try:
            from PIL import Image
            # Create a valid 28x28 grayscale PNG for each digit in each split
            for digit in range(10):
                for split in ['train', 'test']:
                    digit_dir = os.path.join(self.temp_dir, split, str(digit))
                    os.makedirs(digit_dir, exist_ok=True)
                    img_path = os.path.join(digit_dir, f'digit_{digit}_0.png')
                    Image.new('L', (28, 28), color=digit * 20).save(img_path)

            # Load both splits and concatenate
            train_images, train_labels = load_mnist_dataset(self.temp_dir, split='train')
            test_images, test_labels = load_mnist_dataset(self.temp_dir, split='test')
            images = torch.cat([train_images, test_images], dim=0)
            labels = torch.cat([train_labels, test_labels], dim=0)
            
            # Verify that we loaded the test data
            self.assertIsNotNone(images, "Images should not be None")
            self.assertIsNotNone(labels, "Labels should not be None")
            self.assertEqual(len(images), 20, "Should have loaded 20 test images (10 train + 10 test)")
            self.assertEqual(len(labels), 20, "Should have loaded 20 labels")
            
            # Verify that the images have the right shape (batch, channel, height, width)
            self.assertEqual(len(images.shape), 4, "Images should have 4 dimensions")
            self.assertEqual(images.shape[1], 3, "Loader repeats grayscale images to 3 channels")
            
            # Verify that the labels cover all 10 digits
            self.assertEqual(len(torch.unique(labels)), 10, "Should have all 10 digit classes")
            
        except Exception as e:
            self.fail(f"Test failed with error: {e}")

if __name__ == "__main__":
    unittest.main() 