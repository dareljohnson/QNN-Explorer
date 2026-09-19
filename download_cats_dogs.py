import os
import shutil
import requests
import zipfile
from PIL import Image
import io
import random
import sys
from tqdm import tqdm

# Import our utility function to ensure directories exist
try:
    from data.cats_dogs_utils import ensure_dataset_directories
except ImportError:
    # If we can't import, define the function here
    def ensure_dataset_directories():
        """Ensures dataset directories exist."""
        data_dir = 'data/demo_data/cats_dogs'
        train_cat_dir = os.path.join(data_dir, 'train/cat')
        train_dog_dir = os.path.join(data_dir, 'train/dog')
        test_cat_dir = os.path.join(data_dir, 'test/cat')
        test_dog_dir = os.path.join(data_dir, 'test/dog')
        
        for directory in [data_dir, train_cat_dir, train_dog_dir, test_cat_dir, test_dog_dir]:
            os.makedirs(directory, exist_ok=True)
            print(f"Created or verified directory: {directory}")
        
        return True

# Create directories for the data
print("Ensuring dataset directories exist...")
ensure_dataset_directories()
data_dir = 'data/demo_data/cats_dogs'

# Subdirectories after creation
train_cat_dir = os.path.join(data_dir, 'train/cat')
train_dog_dir = os.path.join(data_dir, 'train/dog')
test_cat_dir = os.path.join(data_dir, 'test/cat')
test_dog_dir = os.path.join(data_dir, 'test/dog')

# Check if any images already exist
existing_train_cat = len(os.listdir(train_cat_dir))
existing_train_dog = len(os.listdir(train_dog_dir))
existing_test_cat = len(os.listdir(test_cat_dir))
existing_test_dog = len(os.listdir(test_dog_dir))
total_existing = existing_train_cat + existing_train_dog + existing_test_cat + existing_test_dog

if total_existing > 0:
    print(f"\nFound {total_existing} existing images:")
    if existing_train_cat > 0: print(f" - {existing_train_cat} cat images in train directory")
    if existing_train_dog > 0: print(f" - {existing_train_dog} dog images in train directory")
    if existing_test_cat > 0: print(f" - {existing_test_cat} cat images in test directory")
    if existing_test_dog > 0: print(f" - {existing_test_dog} dog images in test directory")
    
    # Ask if user wants to redownload
    overwrite = input("\nImages already exist. Do you want to redownload and overwrite? (y/n): ").strip().lower()
    if overwrite == 'y':
        print("Continuing with download...")
    else:
        print("Exiting without changing existing dataset.")
        sys.exit(0)

# URL for Microsoft's Cats vs Dogs dataset
dataset_url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
zip_path = os.path.join(data_dir, "kagglecatsanddogs_5340.zip")

# Download the dataset if not already downloaded
if not os.path.exists(zip_path):
    print(f"\nDownloading Cats vs Dogs dataset from {dataset_url}")
    try:
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()  # Raise exception for bad status codes
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        print("Download complete!")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        print("Please check your internet connection and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during download: {e}")
        sys.exit(1)
else:
    print(f"\nUsing existing download at {zip_path}")

# Extract the dataset if needed
extract_dir = os.path.join(data_dir, "extracted")
if not os.path.exists(extract_dir) or len(os.listdir(extract_dir)) == 0:
    os.makedirs(extract_dir, exist_ok=True)
    print("\nExtracting zip file...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in tqdm(zip_ref.namelist(), desc="Extracting"):
                try:
                    zip_ref.extract(member, extract_dir)
                except zipfile.error as e:
                    print(f"Error extracting {member}: {e}")
        print("Extraction complete!")
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid zip file.")
        print("Please delete the file and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)
else:
    print(f"\nUsing existing extracted files at {extract_dir}")

# Process and organize the images
def process_images(source_dir, train_dir, test_dir, class_name, max_train=500, max_test=100):
    """Process images from source directory into train/test splits."""
    print(f"\nProcessing {class_name} images...")
    
    # Get all image files
    image_files = []
    try:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
    except Exception as e:
        print(f"Error finding image files: {e}")
        return 0, 0, 0
    
    if not image_files:
        print(f"Warning: No image files found in {source_dir}")
        return 0, 0, 0
    
    print(f"Found {len(image_files)} {class_name} images to process")
    
    # Shuffle to randomize the split
    random.shuffle(image_files)
    
    # Process and copy images
    train_count = 0
    test_count = 0
    skipped = 0
    
    for i, img_path in enumerate(tqdm(image_files, desc=f"Processing {class_name}")):
        try:
            # Decide if this image goes to train or test
            if train_count < max_train:
                target_dir = train_dir
                target_count = train_count
                train_count += 1
            elif test_count < max_test:
                target_dir = test_dir
                target_count = test_count
                test_count += 1
            else:
                # We have enough images
                break
                
            # Open and verify the image is valid
            img = Image.open(img_path)
            img = img.convert('RGB')  # Convert to RGB if not already
            
            # Save to target directory
            new_filename = f"{class_name}_{target_count:04d}.jpg"
            target_path = os.path.join(target_dir, new_filename)
            img.save(target_path)
            
        except (IOError, OSError) as e:
            print(f"Error opening image {img_path}: {e}")
            skipped += 1
            continue
        except Exception as e:
            print(f"Unexpected error processing {img_path}: {e}")
            skipped += 1
            continue
    
    return train_count, test_count, skipped

# Location of cat and dog folders
cat_dir = os.path.join(extract_dir, "PetImages", "Cat")
dog_dir = os.path.join(extract_dir, "PetImages", "Dog")

# Check if directories exist with different path formats
if not os.path.exists(cat_dir) or not os.path.exists(dog_dir):
    # Try alternative path format
    cat_dir = os.path.join(extract_dir, "PetImages/Cat")
    dog_dir = os.path.join(extract_dir, "PetImages/Dog")

# Try another possible path format
if not os.path.exists(cat_dir) or not os.path.exists(dog_dir):
    # Look for directories named 'Cat' and 'Dog' anywhere in the extract directory
    cat_found = False
    dog_found = False
    
    print("\nSearching for Cat and Dog directories...")
    for root, dirs, files in os.walk(extract_dir):
        for dirname in dirs:
            if dirname.lower() == 'cat':
                cat_dir = os.path.join(root, dirname)
                cat_found = True
                print(f"Found Cat directory: {cat_dir}")
            elif dirname.lower() == 'dog':
                dog_dir = os.path.join(root, dirname)
                dog_found = True
                print(f"Found Dog directory: {dog_dir}")
        
        if cat_found and dog_found:
            break

if not os.path.exists(cat_dir) or not os.path.exists(dog_dir):
    print("\nError: Could not find Cat and Dog directories in the extracted zip.")
    print(f"Looking for: {cat_dir} and {dog_dir}")
    print("Available directories:")
    for root, dirs, files in os.walk(extract_dir):
        if dirs:
            print(f"- {root}: {dirs}")
    
    print("\nPlease check the zip file structure or try downloading again.")
    sys.exit(1)

# Process cat and dog images
print("\nStarting image processing...")
cat_train, cat_test, cat_skipped = process_images(cat_dir, train_cat_dir, test_cat_dir, "cat")
dog_train, dog_test, dog_skipped = process_images(dog_dir, train_dog_dir, test_dog_dir, "dog")

# Final report
print("\nProcessing complete!")
print(f"Cat images: {cat_train} train, {cat_test} test, {cat_skipped} skipped")
print(f"Dog images: {dog_train} train, {dog_test} test, {dog_skipped} skipped")
print(f"Total: {cat_train + dog_train} train, {cat_test + dog_test} test")

# Verify files were created
train_cat_files = os.listdir(train_cat_dir)
train_dog_files = os.listdir(train_dog_dir)
test_cat_files = os.listdir(test_cat_dir)
test_dog_files = os.listdir(test_dog_dir)

print(f"\nVerification:")
print(f"Files in train/cat: {len(train_cat_files)}")
print(f"Files in train/dog: {len(train_dog_files)}")
print(f"Files in test/cat: {len(test_cat_files)}")
print(f"Files in test/dog: {len(test_dog_files)}")

print("\nDataset preparation complete!")

# Final status check
if cat_train == 0 and dog_train == 0:
    print("\n❌ ERROR: No images were processed successfully!")
    print("Please check the dataset structure and try again.")
    sys.exit(1)
elif cat_train < 400 or dog_train < 400:
    print("\n⚠️ WARNING: Fewer images than expected were processed.")
    print(f"Only {cat_train} cat images and {dog_train} dog images were added to training set.")
    print("This may affect model performance, but you can still use the dataset.")
else:
    print("\n✅ SUCCESS: Dataset preparation complete and ready for use!") 