import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info messages
import sys
sys.dont_write_bytecode = True

# Add GPU configuration
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Use the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Found {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("No GPU found, using CPU instead")

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import os.path

# Data loading and preprocessing
def load_data(data_dir="../processed_data", batch_size=16):
    print("\nLoading data...")
    # Load train, dev, test sets
    train_images = np.load(os.path.join(data_dir, "train_images_normalized.npy"))
    train_labels = np.load(os.path.join(data_dir, "train_labels.npy"))
    dev_images = np.load(os.path.join(data_dir, "dev_images_normalized.npy"))
    dev_labels = np.load(os.path.join(data_dir, "dev_labels.npy"))
    test_images = np.load(os.path.join(data_dir, "test_images_normalized.npy"))
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"))
    
    # Resize images to 208x208
    def resize_images(images):
        return tf.image.resize(images, (208, 208))
    
    train_images = resize_images(train_images)
    dev_images = resize_images(dev_images)
    test_images = resize_images(test_images)
    
    print(f"Dataset shapes after resizing:")
    print(f"Train: {train_images.shape} images, {train_labels.shape} labels")
    print(f"Dev:   {dev_images.shape} images, {dev_labels.shape} labels")
    print(f"Test:  {test_images.shape} images, {test_labels.shape} labels")
    
    # Create data generators with optimized pipeline
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))\
        .shuffle(buffer_size=1024)\
        .batch(batch_size)\
        .cache()\
        .prefetch(tf.data.AUTOTUNE)
    
    dev_dataset = tf.data.Dataset.from_tensor_slices((dev_images, dev_labels))\
        .batch(batch_size)\
        .cache()\
        .prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))\
        .batch(batch_size)\
        .cache()
    
    return train_dataset, dev_dataset, test_dataset

# Define the CNN model
def create_model():
    inputs = tf.keras.Input(shape=(208, 208, 3))  # Updated input shape
    
    # Initial dimensionality reduction with strided convolution
    x = tf.keras.layers.Conv2D(8, 3, strides=2, padding='same', activation='relu')(inputs)
    
    # First conv block - using depthwise separable convolutions
    x = tf.keras.layers.SeparableConv2D(8, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    # Second conv block with skip connection
    skip1 = x
    x = tf.keras.layers.SeparableConv2D(12, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, tf.keras.layers.Conv2D(12, 1)(skip1)])
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    # Third conv block
    x = tf.keras.layers.SeparableConv2D(16, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Single dense layer with dropout
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(14, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

# Training setup
def train_model(batch_size=16):
    # Load data as generators
    train_dataset, dev_dataset, test_dataset = load_data(batch_size=batch_size)
    
    print("\nCreating and compiling model...")
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Custom callback for printing
    class PrintEpochResults(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            if epoch == 0:
                print("\nStarting training...")
                print("=" * 80)
        
        def on_epoch_end(self, epoch, logs=None):
            print(f"Epoch {epoch+1:3d}: train_loss: {logs['loss']:.4f}, train_acc: {logs['accuracy']:.4f}, "
                  f"val_loss: {logs['val_loss']:.4f}, val_acc: {logs['val_accuracy']:.4f}")
    
    callbacks = [
        PrintEpochResults(),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
    ]
    
    # Train the model using datasets
    history = model.fit(
        train_dataset,
        epochs=5,
        validation_data=dev_dataset,
        callbacks=callbacks,
        verbose=0  # Turn off default progress bar
    )
    
    # Evaluate on test set using the already created test_dataset
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Save final model
    print("\nSaving final model...")
    model.save('final_model.h5')
    
    return model, history

if __name__ == "__main__":
    print("Starting CNN training pipeline...")
    # Train the model
    model, history = train_model()
    
    print("\nGenerating training plots...")
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

