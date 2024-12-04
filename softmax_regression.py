import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def softmax(z):
    """
    Compute softmax values for each set of scores in z.
    
    Parameters:
        z: numpy array of shape (num_classes, num_samples)
           Contains the raw scores for each class and sample
           
    Returns:
        Softmax probabilities of shape (num_classes, num_samples)
    """
    # Subtract max value to prevent overflow/underflow
    z_shifted = z - np.max(z, axis=0)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=0)

def cross_entropy(y_pred, y_true):
    """
    Calculate cross entropy loss between predicted and true labels.
    
    Parameters:
        y_pred: numpy array of shape (num_classes, num_samples) 
               Contains predicted probabilities
        y_true: numpy array of shape (num_classes, num_samples)
                Contains one-hot encoded true labels
                
    Returns:
        Cross entropy loss (scalar value)
    """
    # Add small epsilon to prevent taking log of 0
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))

def fit_model(x, y, learning_rate=0.01, num_epochs=1000, batch_size=16, lambda_reg=0.01, decay_rate=0.01):
    """
    Parameters:
        x: numpy array of shape (num_features, num_samples)
           Contains the features for each sample
        y: numpy array of shape (num_classes, num_samples)
           Contains one-hot encoded true labels for each sample
        learning_rate: float, learning rate for gradient descent (default: 0.01)
        num_epochs: int, number of epochs to train the model (default: 1000)
        batch_size: int, number of samples per batch (default: 16)
        lambda_reg: float, regularization strength (default: 0.01)
        decay_rate: float, controls how fast learning rate decays (default: 0.01)
                   learning_rate = initial_lr / (1 + decay_rate * epoch)
    """
    num_features = x.shape[0]
    num_classes = y.shape[0]
    num_samples = x.shape[1]
    
    # Initialize weights and parameters
    W = np.random.randn(num_classes, num_features) * 0.01
    b = np.zeros((num_classes, 1))
    initial_lr = learning_rate  # Store initial learning rate
    
    for i in range(num_epochs):
        if i == 0:
            print("First epoch")
        
        # Calculate current learning rate using new decay schedule
        current_lr = initial_lr / (1 + decay_rate * i)
        
        # Shuffle data
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        x = x[:, indices]
        y = y[:, indices]
        
        # Split data into batches - Fix the splitting to maintain the correct dimensions
        num_batches = num_samples // batch_size
        x_batches = [x[:, i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
        y_batches = [y[:, i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
        
        for j in range(num_batches):
            if j == 0 and i == 0:
                print("First batch")
            
            # Randomly select batch_size samples
            x_batch = x_batches[j]
            y_batch = y_batches[j]

            # Forward pass (now using smaller batch)
            z = np.dot(W, x_batch) + b
            y_pred = softmax(z)
            
            # Backward pass with L2 regularization
            dW = (np.dot((y_pred - y_batch), x_batch.T) / batch_size) + (lambda_reg * W)
            db = np.sum(y_pred - y_batch, axis=1, keepdims=True) / batch_size
            
            # Update parameters with current learning rate
            W -= current_lr * dW
            b -= current_lr * db
        
        # Print progress with current learning rate
        if i % 20 == 0:
            y_pred = softmax(np.dot(W, x) + b)
            data_loss = cross_entropy(y_pred, y) / num_samples
            reg_loss = 0.5 * lambda_reg * np.sum(W * W)
            total_loss = data_loss + reg_loss
            accuracy = np.mean(np.argmax(y_pred, axis=0) == np.argmax(y, axis=0))
            print(f"Epoch {i}, Total Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return W, b


def load_data(processed_data_path, test=False):
    """
    Load and preprocess training and development data from .npy files
    
    Parameters:
        train_path: str, path to directory containing training files
        dev_path: str, path to directory containing development files
        
    Returns:
        train_data: tuple of (processed_images, one_hot_labels) for training
        dev_data: tuple of (processed_images, one_hot_labels) for development
    
    Images are first resized from (num_samples, 416, 416, 3) to (num_samples, 64, 64, 3)
    Images are then reshaped from (num_samples, 64, 64, 3) to (num_features, num_samples)
    Labels are converted from (num_samples,) to one-hot encoding (num_classes, num_samples)
    """
    # Load raw data
    train_images = np.load(f"{processed_data_path}/train_images.npy")
    train_labels = np.load(f"{processed_data_path}/train_labels.npy")
    dev_images = np.load(f"{processed_data_path}/dev_images.npy")
    dev_labels = np.load(f"{processed_data_path}/dev_labels.npy")
    if test:
        test_images = np.load(f"{processed_data_path}/test_images.npy")
        test_labels = np.load(f"{processed_data_path}/test_labels.npy")
    
    # Let's resize the images to 64x64x3. 
    train_images_resized = np.array([cv2.resize(img, (64, 64)) for img in train_images])
    dev_images_resized = np.array([cv2.resize(img, (64, 64)) for img in dev_images])
    if test:
        test_images_resized = np.array([cv2.resize(img, (64, 64)) for img in test_images])
    
    # Process images
    # Reshape from (num_samples, 64, 64, 3) to (num_features, num_samples) and normalize
    train_images_processed = (train_images_resized.reshape(train_images_resized.shape[0], -1) / 255.0).T
    dev_images_processed = (dev_images_resized.reshape(dev_images_resized.shape[0], -1) / 255.0).T
    if test:
        test_images_processed = (test_images_resized.reshape(test_images_resized.shape[0], -1) / 255.0).T
    
    # Convert labels to one-hot encoding (num_classes, num_samples)
    num_classes = len(np.unique(train_labels))
    train_labels_one_hot = np.eye(num_classes)[train_labels].T
    dev_labels_one_hot = np.eye(num_classes)[dev_labels].T
    if test:
        test_labels_one_hot = np.eye(num_classes)[test_labels].T
    
    if test:
        return (train_images_processed, train_labels_one_hot), (dev_images_processed, dev_labels_one_hot), (test_images_processed, test_labels_one_hot)
    else:
        return (train_images_processed, train_labels_one_hot), (dev_images_processed, dev_labels_one_hot)

def main_hyperparameter_search(processed_data_path):
    """
    Main function to train and evaluate a softmax regression model.
    
    Parameters:
        train_path: str, path to training data CSV file
        dev_path: str, path to development/validation data CSV file
        
    The data files should have features in all columns except the last one,
    which contains the class labels. The function:
    1. Loads and preprocesses the data
    2. Trains a softmax regression model on the training data
    3. Evaluates the model on both training and validation sets
    4. Prints the classification accuracy for both sets
    """
    print("Loading and preprocessing data...")
    # Load and preprocess data
    (x_train, y_train), (x_val, y_val) = load_data(processed_data_path)
    print(f"Training data shape: {x_train.shape}")
    print(f"Validation data shape: {x_val.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation labels shape: {y_val.shape}")
    
    print("\nTraining model...")
    # Train model
    # Let's try to find the best hyperparameters
    # We will use random search to find the best hyperparameters
    # We will use the validation set to evaluate the model
    
    learning_rates = [10 ** np.random.uniform(-2, -1) for i in range(10)]
    batch_sizes = [8 for i in range(10)]
    lambda_regs = [np.random.uniform(0.001, 0.002) for i in range(10)]
    decay_rates = [np.random.uniform(0.05, 0.15) for i in range(10)]
    
    # Initialize lists to store results
    results = []
    
    for i in range(10):
        W, b = fit_model(x_train, y_train, lambda_reg=lambda_regs[i], 
                        learning_rate=learning_rates[i], num_epochs=100, 
                        batch_size=batch_sizes[i], decay_rate=decay_rates[i])
        
        # Evaluate on validation set
        y_pred_train = softmax(np.dot(W, x_train) + b)
        train_acc = np.mean(np.argmax(y_pred_train, axis=0) == np.argmax(y_train, axis=0))
        y_pred_val = softmax(np.dot(W, x_val) + b)
        val_acc = np.mean(np.argmax(y_pred_val, axis=0) == np.argmax(y_val, axis=0))
        
        # Store results
        results.append({
            'learning_rate': learning_rates[i],
            'batch_size': batch_sizes[i],
            'lambda_reg': lambda_regs[i],
            'decay_rate': decay_rates[i],
            'val_acc': val_acc
        })
        
        print(f"Trial {i+1}/10 - Validation accuracy: {val_acc:.4f}, Training accuracy: {train_acc:.4f}")
    
    # Convert results to numpy arrays for easier plotting
    results_array = np.array([(r['learning_rate'], r['batch_size'], 
                             r['lambda_reg'], r['decay_rate'], r['val_acc']) 
                             for r in results])
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot accuracy vs each hyperparameter
    ax1.semilogx(results_array[:, 0], results_array[:, 4], 'bo')
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('Validation Accuracy')
    
    ax2.semilogx(results_array[:, 1], results_array[:, 4], 'ro')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Validation Accuracy')
    
    ax3.semilogx(results_array[:, 2], results_array[:, 4], 'go')
    ax3.set_xlabel('Lambda (Regularization)')
    ax3.set_ylabel('Validation Accuracy')
    
    ax4.semilogx(results_array[:, 3], results_array[:, 4], 'mo')
    ax4.set_xlabel('Decay Rate')
    ax4.set_ylabel('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('hyperparameter_search_results_2.png')
    
    # Print best model parameters
    best_idx = np.argmax(results_array[:, 4])
    print("\nBest Model Parameters:")
    print(f"Learning Rate: {results_array[best_idx, 0]:.6f}")
    print(f"Batch Size: {int(results_array[best_idx, 1])}")
    print(f"Lambda Regularization: {results_array[best_idx, 2]:.6f}")
    print(f"Decay Rate: {results_array[best_idx, 3]:.6f}")
    print(f"Validation Accuracy: {results_array[best_idx, 4]:.4f}")


def main(processed_data_path):
    print("Loading and preprocessing data...")
    # Load and preprocess data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(processed_data_path, test=True)
    print(f"Training data shape: {x_train.shape}")
    print(f"Validation data shape: {x_val.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation labels shape: {y_val.shape}")
    
    print("\nTraining model...")
    
    W, b = fit_model(x_train, y_train, lambda_reg=0.001833, 
                        learning_rate=0.058062, num_epochs=300, 
                        batch_size=8, decay_rate=0.097080)
        
    # Evaluate on dev/test set
    y_pred_train = softmax(np.dot(W, x_train) + b)
    train_acc = np.mean(np.argmax(y_pred_train, axis=0) == np.argmax(y_train, axis=0))
    y_pred_val = softmax(np.dot(W, x_val) + b)
    val_acc = np.mean(np.argmax(y_pred_val, axis=0) == np.argmax(y_val, axis=0))
    y_pred_test = softmax(np.dot(W, x_test) + b)
    test_acc = np.mean(np.argmax(y_pred_test, axis=0) == np.argmax(y_test, axis=0))
    
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Save the model parameters in a file
    save_dir = os.path.join('weights')
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, 'softmax_regression_params.npz'), W=W, b=b)

if __name__ == "__main__":
    main("processed_data")    