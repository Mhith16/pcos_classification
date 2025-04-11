# visualize_results.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from torchvision import transforms
from PIL import Image

import config
from models import PCOSClassifier
from dataset import get_dataloaders
from visualization import (
    plot_confusion_matrix, plot_roc_curve, 
    plot_precision_recall_curve, plot_training_curves,
    plot_sample_predictions, generate_gradcam, visualize_feature_maps
)

def visualize_training_history(run_id, fold=1):
    """Visualize training history curves."""
    fold_dir = os.path.join(config.OUTPUT_DIR, run_id, f"fold_{fold}")
    vis_dir = os.path.join(fold_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load phase 1 history
    phase1_path = os.path.join(fold_dir, "history.json")
    phase1_history = None
    if os.path.exists(phase1_path):
        with open(phase1_path, 'r') as f:
            phase1_history = json.load(f)
    
    # Load phase 2 history
    phase2_path = os.path.join(fold_dir, "phase2_history.json")
    phase2_history = None
    if os.path.exists(phase2_path):
        with open(phase2_path, 'r') as f:
            phase2_history = json.load(f)
    
    # Combine histories
    combined_history = {}
    if phase1_history:
        combined_history["Phase 1"] = phase1_history
    if phase2_history:
        combined_history["Phase 2"] = phase2_history
    
    if combined_history:
        metrics = ['loss', 'acc', 'auc', 'precision', 'recall']
        
        # Plot all metrics
        for metric in metrics:
            plot_path = os.path.join(vis_dir, f"{metric}_history.png")
            
            plt.figure(figsize=(12, 6))
            
            # Plot each phase
            for phase_name, history in combined_history.items():
                if f'train_{metric}' in history or 'train_accuracy' in history:
                    # Handle different naming conventions
                    train_key = f'train_{metric}' if f'train_{metric}' in history else 'train_accuracy'
                    val_key = f'val_{metric}' if f'val_{metric}' in history else 'val_accuracy'
                    
                    plt.plot(history[train_key], label=f'{phase_name} Train {metric.capitalize()}')
                    plt.plot(history[val_key], label=f'{phase_name} Val {metric.capitalize()}', linestyle='--')
            
            plt.title(f'{metric.capitalize()} over Training')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved {metric} history plot to {plot_path}")

def visualize_model_predictions(run_id, fold=1, num_samples=16):
    """Visualize model predictions with GradCAM on sample images."""
    fold_dir = os.path.join(config.OUTPUT_DIR, run_id, f"fold_{fold}")
    model_dir = os.path.join(fold_dir, "models")
    vis_dir = os.path.join(fold_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load model
    model_path = os.path.join(model_dir, "final_model.pt")
    model = PCOSClassifier()
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    model.eval()
    
    # Get test data
    _, _, test_loader, _ = get_dataloaders()
    
    # Plot sample predictions
    plot_sample_predictions(
        model, 
        test_loader, 
        class_names=['Normal', 'PCOS'], 
        num_samples=num_samples,
        save_path=os.path.join(vis_dir, 'sample_predictions.png'),
        device=config.DEVICE
    )
    
    # Generate GradCAM visualizations for a few samples
    grad_cam_dir = os.path.join(vis_dir, 'gradcam')
    os.makedirs(grad_cam_dir, exist_ok=True)
    
    # Get a few sample images from test loader
    samples = []
    labels = []
    count = 0
    
    for inputs, targets in test_loader:
        for i in range(min(4, len(inputs))):
            samples.append(inputs[i])
            labels.append(targets[i].item())
            count += 1
            if count >= 4:  # Just get 4 samples
                break
        if count >= 4:
            break
    
    # Find a suitable convolutional layer for each backbone
    target_layers = [
        "feature_extractors.0.features.28",  # VGG16 last conv
        "feature_extractors.1.Mixed_7c",     # InceptionV3 last mixed
        "feature_extractors.2.features.denseblock4" # DenseNet last dense block
    ]
    
    # Generate GradCAM for each sample
    for i, (sample, label) in enumerate(zip(samples, labels)):
        for j, layer_name in enumerate(target_layers):
            save_path = os.path.join(grad_cam_dir, f'gradcam_sample{i+1}_backbone{j+1}.png')
            generate_gradcam(
                model, 
                sample.to(config.DEVICE),
                layer_name,
                class_idx=0,  # For binary classification, we can use class 0
                save_path=save_path
            )

def visualize_feature_activations(run_id, fold=1):
    """Visualize feature activations from different layers."""
    fold_dir = os.path.join(config.OUTPUT_DIR, run_id, f"fold_{fold}")
    model_dir = os.path.join(fold_dir, "models")
    vis_dir = os.path.join(fold_dir, "visualizations")
    feature_dir = os.path.join(vis_dir, 'feature_maps')
    os.makedirs(feature_dir, exist_ok=True)
    
    # Load model
    model_path = os.path.join(model_dir, "final_model.pt")
    model = PCOSClassifier()
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    model.eval()
    
    # Get test data
    _, _, test_loader, _ = get_dataloaders()
    
    # Get a sample image
    for inputs, _ in test_loader:
        sample = inputs[0]
        break
    
    # Layers to visualize
    layers = [
        "feature_extractors.0.features.24",  # VGG16 middle layer
        "feature_extractors.1.Mixed_5d",     # InceptionV3 middle layer
        "feature_extractors.2.features.denseblock3" # DenseNet middle layer
    ]
    
    # Visualize feature maps
    for i, layer_name in enumerate(layers):
        save_path = os.path.join(feature_dir, f'feature_maps_backbone{i+1}.png')
        visualize_feature_maps(
            model,
            sample.to(config.DEVICE),
            layer_name,
            num_features=16,
            save_path=save_path
        )

def create_paper_figures(run_id, fold=1):
    """Create publication-ready figures for your research paper."""
    fold_dir = os.path.join(config.OUTPUT_DIR, run_id, f"fold_{fold}")
    eval_dir = os.path.join(fold_dir, "evaluation")
    vis_dir = os.path.join(fold_dir, "visualizations")
    paper_dir = os.path.join(fold_dir, "paper_figures")
    os.makedirs(paper_dir, exist_ok=True)
    
    # Load metrics
    metrics_path = os.path.join(eval_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        import pandas as pd
        metrics = pd.read_csv(metrics_path)
        
        # Create summary table
        fig, ax = plt.figure(figsize=(8, 3)), plt.gca()
        ax.axis('off')
        table = plt.table(
            cellText=metrics.values,
            colLabels=metrics.columns,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        plt.savefig(os.path.join(paper_dir, "metrics_table.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Load confusion matrix, ROC and PR curves from evaluation directory
    for file_name in ['confusion_matrix.png', 'roc_curve.png', 'pr_curve.png']:
        src_path = os.path.join(eval_dir, file_name)
        if os.path.exists(src_path):
            # Copy and improve the plot for the paper
            import shutil
            dst_path = os.path.join(paper_dir, file_name)
            shutil.copy(src_path, dst_path)
    
    # Create model architecture diagram (placeholder)
    plt.figure(figsize=(12, 6))
    plt.text(0.5, 0.5, 'Multi-Backbone Vision Transformer Model Architecture', 
             ha='center', va='center', fontsize=16)
    plt.axis('off')
    plt.savefig(os.path.join(paper_dir, "model_architecture.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created paper figures in {paper_dir}")

def main(run_id, fold=1):
    """Run all visualizations."""
    print(f"Creating visualizations for run {run_id}, fold {fold}")
    
    # 1. Training history curves
    visualize_training_history(run_id, fold)
    
    # 2. Sample predictions and GradCAM
    visualize_model_predictions(run_id, fold)
    
    # 3. Feature activations
    visualize_feature_activations(run_id, fold)
    
    # 4. Publication-ready figures
    create_paper_figures(run_id, fold)
    
    print("All visualizations completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize model results')
    parser.add_argument('--run_id', type=str, required=True, help='Run ID to visualize')
    parser.add_argument('--fold', type=int, default=1, help='Fold number')
    
    args = parser.parse_args()
    main(args.run_id, args.fold)