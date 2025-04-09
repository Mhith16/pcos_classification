"""Visualization functions for PCOS classification with PyTorch."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import cv2
from sklearn.metrics import confusion_matrix
import pandas as pd
from torchvision import transforms
from PIL import Image

def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_roc_curve(fpr, tpr, roc_auc, save_path=None):
    """Plot ROC curve."""
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_precision_recall_curve(precision, recall, pr_auc, save_path=None):
    """Plot precision-recall curve."""
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.fill_between(recall, precision, alpha=0.2, color='green')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_training_curves(history, save_path=None, metrics=None):
    """Plot training and validation curves for specified metrics."""
    if metrics is None:
        metrics = ['loss', 'acc']
    
    # Determine number of phases
    num_phases = len(history)
    
    # Create subplots
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot each phase
        for j, (phase_name, phase_history) in enumerate(history.items()):
            train_metric = None
            val_metric = None
            
            # Find the right keys for this metric
            for key in phase_history.keys():
                if metric in key:
                    if key.startswith('train_'):
                        train_metric = key
                    elif key.startswith('val_'):
                        val_metric = key
                    # Special case for accuracy which might not have train_ prefix
                    elif key == 'acc' or key == 'accuracy':
                        train_metric = key
                    elif key == 'val_acc' or key == 'val_accuracy':
                        val_metric = key
            
            if train_metric and val_metric:
                # Calculate epoch offset for this phase
                if j == 0:
                    offset = 0
                else:
                    # Sum the epochs from previous phases
                    offset = sum(len(history[p][list(history[p].keys())[0]]) for p in list(history.keys())[:j])
                
                # Get epochs for this phase (with offset)
                epochs = range(offset + 1, offset + len(phase_history[train_metric]) + 1)
                
                # Plot training
                ax.plot(
                    epochs, 
                    phase_history[train_metric], 
                    color=colors[j % len(colors)], 
                    label=f'{phase_name} train {metric}'
                )
                
                # Plot validation
                ax.plot(
                    epochs, 
                    phase_history[val_metric], 
                    color=colors[j % len(colors)], 
                    linestyle='--', 
                    label=f'{phase_name} val {metric}'
                )
        
        ax.set_title(f'{metric.capitalize()} over epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_sample_predictions(model, test_loader, class_names, num_samples=16, save_path=None, device='cuda'):
    """Plot sample predictions with true and predicted labels."""
    # Set model to evaluation mode
    model.eval()
    
    # Get a batch of images and their true labels
    images = []
    true_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            batch_images = inputs.cpu().numpy()
            batch_labels = labels.cpu().numpy()
            
            # Add samples from this batch
            for i in range(min(len(batch_images), num_samples - len(images))):
                images.append(batch_images[i])
                true_labels.append(batch_labels[i])
            
            if len(images) >= num_samples:
                break
    
    # Convert to numpy arrays
    images = np.array(images)
    true_labels = np.array(true_labels)
    
    # Get predictions
    with torch.no_grad():
        inputs = torch.tensor(images, dtype=torch.float32).to(device)
        outputs = model(inputs).cpu().numpy().flatten()
    
    pred_labels = (outputs > 0.5).astype(int)
    
    # Plot images with true and predicted labels
    fig, axes = plt.subplots(
        int(np.ceil(num_samples / 4)), 4, 
        figsize=(15, int(np.ceil(num_samples / 4) * 3.5))
    )
    axes = axes.flatten()
    
    # Inverse normalization function to display images
    inv_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    ])
    
    for i, (image, true_label, pred_label, pred_prob) in enumerate(
        zip(images, true_labels, pred_labels, outputs)
    ):
        if i >= num_samples:
            break
            
        ax = axes[i]
        
        # Unnormalize image for display
        img_tensor = torch.tensor(image)
        img_unnorm = inv_normalize(img_tensor).cpu().numpy().transpose(1, 2, 0)
        img_unnorm = np.clip(img_unnorm, 0, 1)
        
        ax.imshow(img_unnorm)
        
        # Determine text color based on correct/incorrect prediction
        color = 'green' if true_label == pred_label else 'red'
        
        # Add labels
        ax.set_title(
            f"True: {class_names[int(true_label)]}\n"
            f"Pred: {class_names[int(pred_label)]} ({pred_prob:.2f})",
            color=color
        )
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def generate_gradcam(model, img_tensor, target_layer_name, class_idx=0, save_path=None):
    """Generate Grad-CAM visualization for the given image and layer."""
    # Put model in evaluation mode
    model.eval()
    
    # Register hooks to get activations and gradients
    activations = {}
    gradients = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    def get_gradient(name):
        def hook(grad):
            gradients[name] = grad.detach()
        return hook
    
    # Find the target layer
    for name, module in model.named_modules():
        if name == target_layer_name:
            module.register_forward_hook(get_activation(target_layer_name))
            break
    
    # Forward pass
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    output = model(img_tensor)
    
    # Get the gradient of the output with respect to the parameters of the model
    model.zero_grad()
    output[0, class_idx].backward()
    
    # Get activations and gradients
    activations_value = activations[target_layer_name]
    if target_layer_name in gradients:
        gradients_value = gradients[target_layer_name]
    else:
        print(f"No gradients found for layer {target_layer_name}!")
        return None
    
    # Global average pooling of gradients
    weights = torch.mean(gradients_value, dim=(2, 3), keepdim=True)
    
    # Weight the channels by gradients
    cam = torch.sum(weights * activations_value, dim=1, keepdim=True)
    
    # Apply ReLU and normalize
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-7)
    
    # Resize CAM to input size
    cam = cam.squeeze().cpu().numpy()
    img = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    
    # Unnormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    # Resize CAM to image size
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap / 255.0
    
    # Combine original image with heatmap
    superimposed = 0.6 * img + 0.4 * heatmap
    superimposed = np.clip(superimposed, 0, 1)
    
    # Plot original and Grad-CAM images
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed)
    plt.title('Grad-CAM')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return superimposed

def visualize_feature_maps(model, img_tensor, layer_name, num_features=16, save_path=None):
    """Visualize feature maps for a specific layer."""
    # Set model to evaluation mode
    model.eval()
    
    # Register hook to get activations
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Find the target layer
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(get_activation(layer_name))
            break
    
    # Forward pass
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        _ = model(img_tensor)
    
    # Get feature maps
    if layer_name not in activations:
        print(f"Layer {layer_name} not found in model!")
        return
    
    feature_maps = activations[layer_name].squeeze().cpu().numpy()
    
    # Plot feature maps
    if len(feature_maps.shape) == 3:  # Shape should be (channels, height, width)
        fig, axes = plt.subplots(
            int(np.ceil(num_features / 4)), 4, 
            figsize=(15, int(np.ceil(num_features / 4) * 3.5))
        )
        axes = axes.flatten()
        
        for i in range(min(num_features, feature_maps.shape[0])):
            fmap = feature_maps[i]
            ax = axes[i]
            ax.imshow(fmap, cmap='viridis')
            ax.set_title(f'Feature {i+1}')
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(min(num_features, feature_maps.shape[0]), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    else:
        print(f"Feature maps shape {feature_maps.shape} not suitable for visualization")