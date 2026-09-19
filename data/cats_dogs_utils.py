import os

def ensure_dataset_directories():
    """
    Ensures that all necessary directories for the cats vs dogs dataset exist.
    Returns True if created or already exists, False if failed.
    """
    # Base directory
    data_dir = 'data/demo_data/cats_dogs'
    
    # Subdirectories
    train_cat_dir = os.path.join(data_dir, 'train/cat')
    train_dog_dir = os.path.join(data_dir, 'train/dog')
    test_cat_dir = os.path.join(data_dir, 'test/cat')
    test_dog_dir = os.path.join(data_dir, 'test/dog')
    
    # Create all directories
    directories = [data_dir, train_cat_dir, train_dog_dir, test_cat_dir, test_dog_dir]
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Created or verified directory: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
            return False
    
    return True

def check_dataset_structure(verbose=True):
    """
    Checks if the cats vs dogs dataset structure exists and is populated.
    Returns a tuple: (structure_exists, num_images_dict)
    """
    # Base directory
    data_dir = 'data/demo_data/cats_dogs'
    
    # Subdirectories
    train_cat_dir = os.path.join(data_dir, 'train/cat')
    train_dog_dir = os.path.join(data_dir, 'train/dog')
    test_cat_dir = os.path.join(data_dir, 'test/cat')
    test_dog_dir = os.path.join(data_dir, 'test/dog')
    
    # Check base directory
    if not os.path.exists(data_dir):
        if verbose:
            print(f"Dataset directory {data_dir} doesn't exist")
        return False, {}
    
    # Check subdirectories and count images
    image_counts = {}
    all_exist = True
    
    directories = {
        'train_cat': train_cat_dir,
        'train_dog': train_dog_dir,
        'test_cat': test_cat_dir,
        'test_dog': test_dog_dir
    }
    
    for key, directory in directories.items():
        if not os.path.exists(directory):
            if verbose:
                print(f"Directory {directory} doesn't exist")
            all_exist = False
            image_counts[key] = 0
        else:
            # Count image files
            files = os.listdir(directory)
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_counts[key] = len(image_files)
            if verbose:
                print(f"Found {len(image_files)} images in {directory}")
    
    # Check if downloads exist
    zip_path = os.path.join(data_dir, "kagglecatsanddogs_5340.zip")
    extract_dir = os.path.join(data_dir, "extracted")
    
    if verbose:
        if os.path.exists(zip_path):
            print(f"Download file exists: {zip_path}")
        else:
            print(f"Download file missing: {zip_path}")
            
        if os.path.exists(extract_dir):
            print(f"Extract directory exists: {extract_dir}")
        else:
            print(f"Extract directory missing: {extract_dir}")
    
    return all_exist, image_counts

if __name__ == "__main__":
    # If this script is run directly, check the dataset structure
    print("Checking cats vs dogs dataset structure...")
    structure_exists, image_counts = check_dataset_structure()
    
    if structure_exists:
        print("\n✅ Dataset structure exists and is ready.")
        print(f"Image counts: {image_counts}")
    else:
        print("\n❌ Dataset structure is incomplete.")
        print("Run download_cats_dogs.py to download and prepare the dataset.")
        
        # Create the directories anyway to help the next step
        ensure_dataset_directories() 