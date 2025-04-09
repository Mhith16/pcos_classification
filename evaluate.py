"""Evaluation script for PCOS classification model using PyTorch."""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, 
    precision_recall_curve, auc, f1_score, 
    matthews_corrcoef, cohen_kappa_score
)
import pandas as pd
from tqdm import tqdm

import config
from dataset import get_dataloaders
from models import PCOSClassifier
from visualization import (
    plot_confusion_matrix, plot_roc_curve, 
    plot_precision_recall_curve, plot_training_curves
)

def evaluate_model(model, test_loader, run_id):
    """Evaluate the model and generate performance metrics."""
    print("Evaluating model...")
    
    # Create directories
    eval_dir = os.path.join(config.OUTPUT_DIR, run_id, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize arrays to store predictions and ground truth
    all_outputs = []
    all_targets = []
    
    # Get predictions
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(config.DEVICE)
            outputs = model(inputs)
            
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all batches
    all_outputs = torch.cat(all_outputs, dim=0).numpy().flatten()
    all_targets = torch.cat(all_targets, dim=0).numpy().flatten()
    
    # Binary predictions
    y_pred = (all_outputs > 0.5).astype(int)
    y_true = all_targets
    
    # Calculate evaluation metrics
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = float(np.mean(y_pred == y_true))
    metrics['loss'] = float(np.mean(-y_true * np.log(all_outputs + 1e-7) - (1 - y_true) * np.log(1 - all_outputs + 1e-7)))
    
    # AUC
    fpr, tpr, _ = roc_curve(y_true, all_outputs)
    metrics['auc'] = float(auc(fpr, tpr))
    
    # Precision and recall
    cm = confusion_matrix(y_true, y_pred)
    metrics['precision'] = float(cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0)
    metrics['recall'] = float(cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0)
    
    # Additional metrics
    metrics['f1_score'] = float(f1_score(y_true, y_pred))
    metrics['specificity'] = float(cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0)
    metrics['matthews_corrcoef'] = float(matthews_corrcoef(y_true, y_pred))
    metrics['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(eval_dir, 'metrics.csv'), index=False)
    
    # Generate confusion matrix
    plot_confusion_matrix(cm, ['Normal', 'PCOS'], os.path.join(eval_dir, 'confusion_matrix.png'))
    
    # Generate ROC curve
    plot_roc_curve(fpr, tpr, metrics['auc'], os.path.join(eval_dir, 'roc_curve.png'))
    
    # Generate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, all_outputs)
    pr_auc = auc(recall, precision)
    plot_precision_recall_curve(precision, recall, pr_auc, os.path.join(eval_dir, 'pr_curve.png'))
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=['Normal', 'PCOS'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(eval_dir, 'classification_report.csv'))
    
    # Print metrics
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Matthews Correlation Coefficient: {metrics['matthews_corrcoef']:.4f}")
    print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    
    return metrics

def evaluate_training_history(history, run_id):
    """Evaluate and visualize training history."""
    eval_dir = os.path.join(config.OUTPUT_DIR, run_id, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Plot training curves for both phases if applicable
    if isinstance(history, dict) and 'phase1' in history and 'phase2' in history:
        plot_training_curves(
            history, 
            os.path.join(eval_dir, 'training_curves.png'),
            metrics=['loss', 'acc', 'auc', 'precision', 'recall']
        )
    else:
        plot_training_curves(
            {'single_phase': history}, 
            os.path.join(eval_dir, 'training_curves.png'),
            metrics=['loss', 'acc', 'auc', 'precision', 'recall']
        )

def evaluate_cv_results(cv_results, run_id):
    """Evaluate and visualize cross-validation results."""
    cv_dir = os.path.join(config.OUTPUT_DIR, run_id, "cv_evaluation")
    os.makedirs(cv_dir, exist_ok=True)
    
    # Extract metrics from each fold
    fold_metrics = [result['metrics'] for result in cv_results]
    
    # Create a DataFrame for easy analysis
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df['fold'] = [result['fold'] for result in cv_results]
    
    # Calculate mean and std for each metric
    mean_metrics = metrics_df.drop('fold', axis=1).mean()
    std_metrics = metrics_df.drop('fold', axis=1).std()
    
    # Save metrics summary
    summary = pd.DataFrame({
        'mean': mean_metrics,
        'std': std_metrics
    })
    summary.to_csv(os.path.join(cv_dir, 'cv_metrics_summary.csv'))
    
    # Plot metrics across folds
    plt.figure(figsize=(12, 8))
    for column in metrics_df.columns:
        if column != 'fold':
            plt.plot(metrics_df['fold'], metrics_df[column], 'o-', label=column)
    
    plt.xlabel('Fold')
    plt.ylabel('Metric Value')
    plt.title('Metrics Across Cross-Validation Folds')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(cv_dir, 'cv_metrics.png'), dpi=300)
    plt.close()
    
    # Print CV summary
    print("\nCross-Validation Results:")
    for metric in ['accuracy', 'auc', 'precision', 'recall']:
        print(f"{metric.capitalize()}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

if __name__ == "__main__":
    # Load the most recent model
    run_dirs = [d for d in os.listdir(config.OUTPUT_DIR) if os.path.isdir(os.path.join(config.OUTPUT_DIR, d))]
    if not run_dirs:
        print("No training runs found!")
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
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    
    # Load test data
    _, _, test_loader, _ = get_dataloaders()
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, latest_run)
    
    # Evaluate training history
    if os.path.exists(os.path.join(config.OUTPUT_DIR, latest_run, "history.json")):
        with open(os.path.join(config.OUTPUT_DIR, latest_run, "history.json"), 'r') as f:
            history = json.load(f)
        evaluate_training_history(history, latest_run)
    
    # Check if this is a CV run
    if "cv_" in latest_run:
        cv_results_path = os.path.join(config.OUTPUT_DIR, latest_run, "cv_results.json")
        if os.path.exists(cv_results_path):
            with open(cv_results_path, 'r') as f:
                cv_results = json.load(f)
            evaluate_cv_results(cv_results, latest_run)