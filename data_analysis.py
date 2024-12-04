import numpy as np
import matplotlib.pyplot as plt

def load_data(processed_data_path):
    train_images = np.load(f"{processed_data_path}/train_images.npy")
    train_labels = np.load(f"{processed_data_path}/train_labels.npy")
    dev_images = np.load(f"{processed_data_path}/dev_images.npy")
    dev_labels = np.load(f"{processed_data_path}/dev_labels.npy")
    test_images = np.load(f"{processed_data_path}/test_images.npy")
    test_labels = np.load(f"{processed_data_path}/test_labels.npy")
    
    return train_images, train_labels, dev_images, dev_labels, test_images, test_labels

def analyze_class_distribution(labels, dataset_name=""):
    """
    Analyze and print the distribution of examples across different classes
    
    Args:
        labels: numpy array of labels
        dataset_name: string to identify which dataset is being analyzed
    """
    unique_classes, counts = np.unique(labels, return_counts=True)
    print(f"\nClass distribution for {dataset_name} dataset:")
    print("-" * 40)
    for class_label, count in zip(unique_classes, counts):
        print(f"Class {class_label}: {count} examples ({count/len(labels)*100:.2f}%)")
    print(f"Total examples: {len(labels)}")

def visualize_class_examples(images, labels, dataset_name="", num_examples=5):
    """
    Display multiple random example images for each class in the dataset
    
    Args:
        images: numpy array of images
        labels: numpy array of labels
        dataset_name: string to identify which dataset is being visualized
        num_examples: number of random examples to show per class
    """
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    
    plt.figure(figsize=(15, 2*num_examples))
    plt.suptitle(f'Random Example Images from Each Class - {dataset_name} Dataset')
    
    for class_idx, class_label in enumerate(unique_classes):
        # Find all images of this class
        class_indices = np.where(labels == class_label)[0]
        # Randomly sample num_examples images
        selected_indices = np.random.choice(class_indices, size=min(num_examples, len(class_indices)), replace=False)
        
        for example_idx, image_idx in enumerate(selected_indices):
            plt.subplot(num_examples, n_classes, example_idx * n_classes + class_idx + 1)
            plt.imshow(images[image_idx], cmap='gray')
            if example_idx == 0:
                plt.title(f'Class {class_label}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Replace with your actual data path
    data_path = "processed_data"
    train_images, train_labels, dev_images, dev_labels, test_images, test_labels = load_data(data_path)
    
    # Analyze distribution for each dataset
    analyze_class_distribution(train_labels, "Training")
    analyze_class_distribution(dev_labels, "Development")
    analyze_class_distribution(test_labels, "Test")
    
    # Visualize multiple random examples from each class
    visualize_class_examples(train_images, train_labels, "Training", num_examples=5)


