import os
import numpy as np
from PIL import Image
import glob

def load_dataset(base_path, split):
    """
    Load images and labels from the specified split (train/dev/valid), excluding class 2.
    
    Parameters:
        base_path: str, path to dataset_raw/car directory
        split: str, one of ['train', 'dev', 'valid']
        
    Returns:
        images: numpy array of shape (n_samples, height, width, channels)
        labels: numpy array of shape (n_samples,)
    """
    # Construct paths
    images_path = os.path.join(base_path, split, 'images', '*.jpg')
    labels_path = os.path.join(base_path, split, 'labels', '*.txt')
    
    # Get sorted lists of file paths
    image_files = sorted(glob.glob(images_path))
    label_files = sorted(glob.glob(labels_path))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {images_path}")
    
    # Read first image to get dimensions
    sample_image = np.array(Image.open(image_files[0]))
    img_height, img_width, channels = sample_image.shape
    
    # Initialize arrays
    images = np.zeros((len(image_files), img_height, img_width, channels), dtype=np.uint8)
    labels = np.zeros(len(label_files), dtype=np.int32)
    
    # Load images
    for i, img_path in enumerate(image_files):
        images[i] = np.array(Image.open(img_path))
    
    # Load labels
    for i, label_path in enumerate(label_files):
        with open(label_path, 'r') as f:
            content = f.read().strip()
            number = ''
            for char in content:
                if char.isnumeric():
                    number += char
                elif number:
                    break
            if number:
                labels[i] = int(number)
    
    # Filter out class 2 and shift higher classes down
    keep_mask = labels != 2
    images = images[keep_mask]
    labels = labels[keep_mask]
    
    # Shift classes above 2 down by 1 to maintain continuous numbering
    labels[labels > 2] -= 1
    
    return images, labels

def main():
    # Adjust this path according to your directory structure
    base_path = os.path.join('raw_data', 'car')
    
    # Load all splits
    splits = ['train', 'dev', 'test']
    datasets = {}
    
    for split in splits:
        print(f"Loading {split} set...")
        images, labels = load_dataset(base_path, split)
        datasets[split] = {
            'images': images,
            'labels': labels
        }
        print(f"{split} set:")
        print(f"- Number of samples: {len(images)}")
        print(f"- Images array shape: {images.shape}")
        print(f"- Labels array shape: {labels.shape}")
        print()
    
    # Optional: save as numpy arrays
    save_dir = os.path.join('processed_data')
    os.makedirs(save_dir, exist_ok=True)
    
    for split in splits:
        np.save(os.path.join(save_dir, f'{split}_images.npy'), datasets[split]['images'])
        np.save(os.path.join(save_dir, f'{split}_labels.npy'), datasets[split]['labels'])

    
if __name__ == "__main__":
    main()
