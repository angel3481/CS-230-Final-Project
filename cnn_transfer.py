import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import TensorDataset
import os
import matplotlib.pyplot as plt
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms for the training data
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224 images
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(20),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the pre-trained ResNet50 model
model = models.resnet50(pretrained=True)

# First freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last two layers (layer3 and layer4)
for param in model.layer3.parameters():
    param.requires_grad = True

for param in model.layer4.parameters():
    param.requires_grad = True

# Modify the final fully connected layer for 14 classes
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 14)  # 14 classes
)

# Move model to device
model = model.to(device)

# Load and prepare the data
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, epoch=0):
        self.images = images
        self.labels = labels
        self.epoch = epoch
        
    def __len__(self):
        return len(self.labels)
    
    def get_transforms_for_epoch(self, epoch):
        # Change transformations every 5 epochs
        epoch_mod = (epoch // 5) % 4  # Create 4 different transformation combinations
        
        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        if epoch_mod == 0:
            # Geometric transformations
            augment_transforms = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(20),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
            ]
        elif epoch_mod == 1:
            # Color transformations
            augment_transforms = [
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomGrayscale(p=0.1)
            ]
        elif epoch_mod == 2:
            # Perspective and flip transformations
            augment_transforms = [
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
            ]
        else:
            # Combined transformations
            augment_transforms = [
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.3)
            ]
        
        return transforms.Compose(augment_transforms + base_transforms)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert numpy array to PIL Image for transformations
        image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Get transforms for current epoch
        transform = self.get_transforms_for_epoch(self.epoch)
        image = transform(image)
        
        return image, label

class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert numpy array to PIL Image
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = self.transform(image)
        
        return image, label

def load_data(split='train', epoch=0):
    # Load numpy arrays
    images = np.load(f'../processed_data/{split}_images.npy')
    labels = np.load(f'../processed_data/{split}_labels.npy')
    
    # Convert labels to torch tensors
    labels = torch.LongTensor(labels)
    
    if split == 'train':
        dataset = AugmentedDataset(images, labels, epoch=epoch)
    else:
        # For dev and test, use EvalDataset with only resize and normalize
        dataset = EvalDataset(images, labels)
    
    loader = DataLoader(dataset, 
                       batch_size=32, 
                       shuffle=(split == 'train'),
                       num_workers=2)
    
    return loader

# Load all datasets
train_loader = load_data('train')
dev_loader = load_data('dev')
test_loader = load_data('test')

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': model.layer3.parameters(), 'lr': 0.00001},  # 10x smaller
    {'params': model.layer4.parameters(), 'lr': 0.00001},  # 10x smaller
    {'params': model.fc.parameters(), 'lr': 0.0001}       # Original learning rate
], lr=0.0001)  # This becomes the default learning rate

# Training loop
num_epochs = 10

def evaluate(loader, split='dev'):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():  # No need to track gradients
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(loader)
    print(f'{split.capitalize()} Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}')
    return accuracy, avg_loss, all_predictions, all_labels

def plot_metrics(train_losses, dev_losses, train_accs, dev_accs, save_path='training_metrics.png'):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, dev_losses, 'r-', label='Dev Loss')
    plt.title('Training and Dev Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, dev_accs, 'r-', label='Dev Accuracy')
    plt.title('Training and Dev Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_model(start_epoch=0, load_checkpoint=False):
    # Initialize lists to store metrics
    train_losses = []
    dev_losses = []
    train_accs = []
    dev_accs = []
    
    if load_checkpoint and os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
        print("Loaded previous best model checkpoint")
    
    best_dev_acc = 0
    if load_checkpoint:
        dev_acc, dev_loss, _, _ = evaluate(dev_loader, 'dev')
        best_dev_acc = dev_acc
        print(f'Initial Dev Accuracy: {best_dev_acc:.2f}%')
    
    # Load initial training data
    train_loader = load_data('train', epoch=start_epoch)
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # Reload training data only every 5 epochs
        if epoch % 5 == 0 and epoch != start_epoch:
            train_loader = load_data('train', epoch=epoch)
            print(f"Reloading training data with new transformations at epoch {epoch}")
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Evaluation phase
        dev_acc, dev_loss, _, _ = evaluate(dev_loader, 'dev')
        
        # Store metrics
        train_losses.append(epoch_loss)
        dev_losses.append(dev_loss)
        train_accs.append(train_acc)
        dev_accs.append(dev_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]:')
        print(f'  Training Loss: {epoch_loss:.4f}')
        print(f'  Training Accuracy: {train_acc:.2f}%')
        
        # Save best model
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'  New best model saved! Dev Accuracy: {dev_acc:.2f}%')
        print()
        
        # Plot metrics after each epoch
        plot_metrics(train_losses, dev_losses, train_accs, dev_accs)
    
    return train_losses, dev_losses, train_accs, dev_accs

if __name__ == "__main__":
    # To continue training from a checkpoint:
    start_epoch = 0  # Your previous training ended at epoch 10
    train_losses, dev_losses, train_accs, dev_accs = train_model(start_epoch=start_epoch, load_checkpoint=False)
    
    # Final evaluation on test set
    print("Final Evaluation:")
    test_acc, test_loss, test_preds, test_labels = evaluate(test_loader, 'test')
