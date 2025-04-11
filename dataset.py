"""PyTorch Dataset and DataLoaders for PCOS classification."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from torchvision import transforms
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split, KFold
import config

class PCOSDataset(Dataset):
    """Dataset class for PCOS classification."""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths (list): List of image file paths
            labels (list): List of labels (0 for normal, 1 for PCOS)
            transform (callable, optional): Transform to be applied to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

def get_transforms():
    """Get transforms for training and validation/testing."""
    # Training transformations with augmentation
    train_transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/test transformations (only resize and normalize)
    val_test_transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

def load_data():
    """Load and prepare dataset from the PCOS directory."""
    print("Loading dataset...")
    
    # Get all image paths and labels
    infected_dir = os.path.join(config.DATA_DIR, "infected")
    noninfected_dir = os.path.join(config.DATA_DIR, "noninfected")
    
    infected_images = [os.path.join(infected_dir, img) for img in os.listdir(infected_dir) 
                      if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    noninfected_images = [os.path.join(noninfected_dir, img) for img in os.listdir(noninfected_dir) 
                         if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    all_images = infected_images + noninfected_images
    labels = [1] * len(infected_images) + [0] * len(noninfected_images)
    
    # Convert to numpy arrays
    images = np.array(all_images)
    labels = np.array(labels)
    
    # Split into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images, labels, test_size=config.TEST_SPLIT, random_state=config.RANDOM_SEED, stratify=labels
    )
    
    # Split train+val into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=config.VALIDATION_SPLIT/(1-config.TEST_SPLIT),
        random_state=config.RANDOM_SEED, 
        stratify=y_train_val
    )
    
    print(f"Dataset loaded: {len(X_train)} training, {len(X_val)} validation, {len(X_test)} test images")
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test),
        'all_data': (images, labels)
    }

def get_dataloaders():
    """Create PyTorch DataLoaders for training, validation and testing."""
    # Get data splits
    data = load_data()
    
    # Get transforms
    train_transform, val_test_transform = get_transforms()
    
    # Create datasets
    train_dataset = PCOSDataset(
        data['train'][0], data['train'][1], transform=train_transform
    )
    
    val_dataset = PCOSDataset(
        data['val'][0], data['val'][1], transform=val_test_transform
    )
    
    test_dataset = PCOSDataset(
        data['test'][0], data['test'][1], transform=val_test_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader, data

def get_cross_validation_folds():
    """Create cross-validation folds for robust evaluation."""
    data = load_data()
    X, y = data['all_data']
    
    # Get transforms
    train_transform, val_test_transform = get_transforms()
    
    # Initialize KFold
    kf = KFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    
    # Create train and validation DataLoaders for each fold
    fold_loaders = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Create datasets
        train_dataset = PCOSDataset(
            X_train_fold, y_train_fold, transform=train_transform
        )
        
        val_dataset = PCOSDataset(
            X_val_fold, y_val_fold, transform=val_test_transform
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        fold_loaders.append({
            'fold': fold_idx + 1,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset)
        })
    
    return fold_loaders