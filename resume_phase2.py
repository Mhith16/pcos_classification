"""Script to resume training from Phase 1 and continue with Phase 2 (fine-tuning)."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import json
import argparse
from datetime import datetime

import config
from models import PCOSClassifier
from dataset import get_dataloaders, PCOSDataset
from utils import AverageMeter, EarlyStopping

def validate(model, val_loader, criterion, device):
    """Validate the model on the validation set."""
    model.eval()
    losses = AverageMeter()
    accuracy = AverageMeter()
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc='Validation'):
            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            
            # Calculate accuracy
            preds = (outputs > 0.5).float()
            acc = (preds == targets.unsqueeze(1)).float().mean()
            
            # Update metrics
            losses.update(loss.item(), inputs.size(0))
            accuracy.update(acc.item(), inputs.size(0))
            
            # Save outputs and targets for additional metrics
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all outputs and targets
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate AUC, precision, recall
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    
    all_outputs_np = all_outputs.numpy().flatten()
    all_targets_np = all_targets.numpy().flatten()
    all_preds_np = (all_outputs_np > 0.5).astype(float)
    
    auc = roc_auc_score(all_targets_np, all_outputs_np)
    precision = precision_score(all_targets_np, all_preds_np, zero_division=0)
    recall = recall_score(all_targets_np, all_preds_np, zero_division=0)
    
    return {
        'loss': losses.avg, 
        'accuracy': accuracy.avg,
        'auc': auc,
        'precision': precision,
        'recall': recall
    }

def train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=1):
    """Train the model for one epoch with gradient accumulation to save memory."""
    model.train()
    losses = AverageMeter()
    accuracy = AverageMeter()
    
    # Create progress bar
    pbar = tqdm(train_loader, desc='Training')
    
    optimizer.zero_grad()  # Zero gradients once at the beginning
    
    for i, (inputs, targets) in enumerate(pbar):
        # Move to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights only every accumulation_steps batches
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()
        
        # Calculate accuracy
        preds = (outputs > 0.5).float()
        acc = (preds == targets.unsqueeze(1)).float().mean()
        
        # Update metrics (use the unscaled loss for display)
        losses.update(loss.item() * accumulation_steps, inputs.size(0))
        accuracy.update(acc.item(), inputs.size(0))
        
        # Update progress bar
        pbar.set_postfix({'loss': losses.avg, 'acc': accuracy.avg})
    
    return {'loss': losses.avg, 'accuracy': accuracy.avg}

def selective_unfreeze(model):
    """Selectively unfreeze only a portion of one backbone to save memory."""
    print("Selectively unfreezing backbone layers...")
    
    # Only unfreeze the final layers of DenseNet (backbone 2) as it's often more effective
    densenet_idx = 2  # Assuming DenseNet is the third backbone
    
    # Get the model to unfreeze
    if len(model.feature_extractors) > densenet_idx:
        extractor = model.feature_extractors[densenet_idx]
        
        # Get all parameters
        all_params = list(extractor.parameters())
        num_params = len(all_params)
        
        # Calculate how many parameters to unfreeze (10% of the model)
        unfreeze_percentage = 0.1
        num_to_unfreeze = int(num_params * unfreeze_percentage)
        
        # Unfreeze the last X% of parameters
        for param in all_params[-num_to_unfreeze:]:
            param.requires_grad = True
        
        # Count number of unfrozen parameters
        unfrozen_count = sum(p.requires_grad for p in extractor.parameters())
        total_count = len(all_params)
        print(f"Unfrozen {unfrozen_count}/{total_count} parameters in DenseNet backbone")
    else:
        print("DenseNet backbone not found at expected index")
    
    # Always unfreeze batch norm layers which help with training but don't add many parameters
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = True

def resume_phase2(run_id, fold=1, epochs=10, batch_size=8, accumulation_steps=4):
    """Resume training from Phase 1 and continue with Phase 2 fine-tuning."""
    
    # If run_id isn't provided, find the most recent run
    if run_id is None:
        run_dirs = [d for d in os.listdir(config.OUTPUT_DIR) 
                   if os.path.isdir(os.path.join(config.OUTPUT_DIR, d)) and d.startswith('cv_run_')]
        if not run_dirs:
            print("No previous runs found!")
            return
        run_id = sorted(run_dirs)[-1]
    
    fold_dir = os.path.join(config.OUTPUT_DIR, run_id, f"fold_{fold}")
    model_dir = os.path.join(fold_dir, "models")
    
    # Make sure directories exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Try to find best model from Phase 1
    model_path = os.path.join(fold_dir, "best_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "phase1_model.pt")
        if not os.path.exists(model_path):
            print(f"No model found at {model_path}")
            return
    
    print(f"Resuming from model: {model_path}")
    
    # Load model
    model = PCOSClassifier()
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    
    # Get original data loaders
    train_loader, val_loader, _, _ = get_dataloaders()
    
    # Create smaller batch size loaders for Phase 2
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    phase2_train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,  # Smaller batch size
        shuffle=True, 
        num_workers=2,  # Reduced workers to avoid warning
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    phase2_val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,  # Smaller batch size
        shuffle=False, 
        num_workers=2,  # Reduced workers
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Selectively unfreeze backbone layers
    selective_unfreeze(model)
    
    # Define loss function
    criterion = nn.BCELoss()
    
    # Define optimizer with smaller learning rate for fine-tuning
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config.FINE_TUNING_LR
    )
    
    # Define learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config.LR_REDUCTION_FACTOR, 
        patience=config.LR_REDUCTION_PATIENCE, 
        verbose=True
    )
    
    # Define early stopping
    best_model_path = os.path.join(fold_dir, 'best_model_phase2.pt')
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        verbose=True,
        path=best_model_path
    )
    
    # Initialize history dictionary for Phase 2
    history = {
        'train_loss': [], 
        'train_acc': [], 
        'val_loss': [], 
        'val_acc': [], 
        'val_auc': [], 
        'val_precision': [], 
        'val_recall': []
    }
    
    print(f"\nStarting Phase 2 training (Fine-tuning) for {epochs} epochs")
    print(f"Batch size: {batch_size}, Gradient accumulation steps: {accumulation_steps}")
    
    # Training loop for Phase 2
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train for one epoch
        train_metrics = train_epoch(
            model, 
            phase2_train_loader, 
            criterion, 
            optimizer, 
            config.DEVICE,
            accumulation_steps=accumulation_steps
        )
        
        # Validate
        val_metrics = validate(model, phase2_val_loader, criterion, config.DEVICE)
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        
        # Check early stopping
        early_stopping(val_metrics['loss'], model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    
    # Load the best model from Phase 2
    model.load_state_dict(torch.load(best_model_path))
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(fold_dir, 'phase2_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Saved training history to {history_path}")
    
    # Final evaluation
    print("\nFinal evaluation on validation set:")
    final_metrics = validate(model, val_loader, criterion, config.DEVICE)
    
    for key, value in final_metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")
    
    return model, history, final_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resume Phase 2 training for PCOS classification')
    parser.add_argument('--run_id', type=str, default=None, help='Run ID to resume from')
    parser.add_argument('--fold', type=int, default=1, help='Fold number')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for Phase 2')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for Phase 2')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    
    args = parser.parse_args()
    
    resume_phase2(
        run_id=args.run_id,
        fold=args.fold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps
    )