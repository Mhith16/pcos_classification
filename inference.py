"""Inference script for PCOS classification using PyTorch."""

import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from torchvision import transforms

import config
from models import PCOSClassifier
from visualization import generate_gradcam

def preprocess_image(image_path):
    """Preprocess a single image for inference."""
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    
    return img_tensor

def predict_image(model, image_path, device, visualize=False):
    """Predict class for a single image."""
    # Preprocess image
    img_tensor = preprocess_image(image_path)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get prediction
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        output = model(img_tensor.unsqueeze(0))
        pred_prob = output.item()
        pred_class = 'PCOS' if pred_prob > 0.5 else 'Normal'
    
    # Visualize if requested
    if visualize:
        # Unnormalize image for display
        inv_normalize = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
        ])
        
        img_unnorm = inv_normalize(img_tensor).cpu().numpy().transpose(1, 2, 0)
        img_unnorm = np.clip(img_unnorm, 0, 1)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img_unnorm)
        plt.title(f'Prediction: {pred_class} ({pred_prob:.4f})')
        plt.axis('off')
        plt.show()
        
        # Generate Grad-CAM
        # Find a suitable convolutional layer for Grad-CAM
        target_layer = None
        for name, module in model.feature_extractors[0].named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = f"feature_extractors.0.{name}"
        
        if target_layer:
            generate_gradcam(model, img_tensor, target_layer)
    
    return {
        'class': pred_class,
        'probability': float(pred_prob),
        'image_path': image_path
    }

def predict_batch(model, image_paths, device, output_dir=None):
    """Predict classes for a batch of images."""
    results = []
    
    for image_path in image_paths:
        result = predict_image(model, image_path, device, visualize=False)
        results.append(result)
        
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Prediction: {result['class']} (Probability: {result['probability']:.4f})")
        print("-" * 50)
    
    # Save results if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save text results
        with open(os.path.join(output_dir, 'predictions.txt'), 'w') as f:
            for result in results:
                f.write(f"Image: {os.path.basename(result['image_path'])}\n")
                f.write(f"Prediction: {result['class']} (Probability: {result['probability']:.4f})\n")
                f.write("-" * 50 + "\n")
        
        # Create a visualization of predictions
        num_images = min(16, len(image_paths))
        fig, axes = plt.subplots(
            int(np.ceil(num_images / 4)), 4, 
            figsize=(15, int(np.ceil(num_images / 4) * 3.5))
        )
        axes = axes.flatten()
        
        for i, result in enumerate(results[:num_images]):
            # Load and transform image
            img_tensor = preprocess_image(result['image_path'])
            
            # Unnormalize for display
            inv_normalize = transforms.Compose([
                transforms.Normalize(
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                    std=[1/0.229, 1/0.224, 1/0.225]
                )
            ])
            
            img_unnorm = inv_normalize(img_tensor).cpu().numpy().transpose(1, 2, 0)
            img_unnorm = np.clip(img_unnorm, 0, 1)
            
            axes[i].imshow(img_unnorm)
            
            # Determine color based on probability
            color = 'green' if result['probability'] > 0.8 or result['probability'] < 0.2 else 'orange'
            
            axes[i].set_title(
                f"{result['class']} ({result['probability']:.2f})",
                color=color
            )
            axes[i].axis('off')
        
        # Hide empty subplots
        for i in range(num_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'prediction_summary.png'), dpi=300)
        plt.close()
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PCOS Classification Inference')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--dir', type=str, help='Path to directory with images')
    parser.add_argument('--output', type=str, help='Path to output directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    
    args = parser.parse_args()
    
    # Set device
    device = config.DEVICE
    
    # Load model
    if args.model:
        model_path = args.model
    else:
        # Try to load the latest model
        run_dirs = [d for d in os.listdir(config.OUTPUT_DIR) if os.path.isdir(os.path.join(config.OUTPUT_DIR, d))]
        if not run_dirs:
            print("No trained models found!")
            exit(1)
        
        latest_run = sorted(run_dirs)[-1]
        
        # Check if it's a CV run
        if "cv_" in latest_run:
            # For CV runs, find the best fold
            cv_results_path = os.path.join(config.OUTPUT_DIR, latest_run, "cv_results.json")
            if os.path.exists(cv_results_path):
                import json
                with open(cv_results_path, 'r') as f:
                    cv_results = json.load(f)
                
                # Find the best fold based on AUC
                best_fold_idx = np.argmax([result['metrics']['auc'] for result in cv_results])
                best_fold = cv_results[best_fold_idx]['fold']
                model_path = os.path.join(config.OUTPUT_DIR, latest_run, f"fold_{best_fold}", "models", "final_model.pt")
                
                # If the model file doesn't exist, try the best_model.pt
                if not os.path.exists(model_path):
                    model_path = os.path.join(config.OUTPUT_DIR, latest_run, f"fold_{best_fold}", "best_model.pt")
            else:
                # If no cv_results.json, just use the first fold
                model_path = os.path.join(config.OUTPUT_DIR, latest_run, "fold_1", "models", "final_model.pt")
                if not os.path.exists(model_path):
                    model_path = os.path.join(config.OUTPUT_DIR, latest_run, "fold_1", "best_model.pt")
        else:
            # For standard runs
            model_path = os.path.join(config.OUTPUT_DIR, latest_run, "models", "final_model.pt")
            if not os.path.exists(model_path):
                model_path = os.path.join(config.OUTPUT_DIR, latest_run, "best_model.pt")
    
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        exit(1)
    
    print(f"Loading model from {model_path}")
    
    # Load model
    model = PCOSClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Process single image or directory
    if args.image:
        result = predict_image(model, args.image, device, visualize=args.visualize)
        print(f"Prediction: {result['class']} (Probability: {result['probability']:.4f})")
    
    elif args.dir:
        # Get all image files from directory
        image_paths = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_paths.extend([
                os.path.join(args.dir, f) for f in os.listdir(args.dir) 
                if f.lower().endswith(ext)
            ])
        
        if not image_paths:
            print(f"No images found in {args.dir}")
            exit(1)
        
        predict_batch(model, image_paths, device, output_dir=args.output)
    
    else:
        print("Please provide either --image or --dir argument")