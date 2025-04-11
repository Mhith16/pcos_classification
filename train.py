"""Training script for PCOS classification model using PyTorch."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

import config
from dataset import get_dataloaders, get_cross_validation_folds
from models import PCOSClassifier
from utils import create_run_id, save_config, EarlyStopping, AverageMeter, set_seed

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    losses = AverageMeter()
    accuracy = AverageMeter()
    
    # Create progress bar
    pbar = tqdm(train_loader, desc='Training')
    # print(config.DEVICE)
    
    for inputs, targets in pbar:
        # Move to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        preds = (outputs > 0.5).float()
        acc = (preds == targets.unsqueeze(1)).float().mean()
        
        # Update metrics
        losses.update(loss.item(), inputs.size(0))
        accuracy.update(acc.item(), inputs.size(0))
        
        # Update progress bar
        pbar.set_postfix({'loss': losses.avg, 'acc': accuracy.avg})
    
    return {'loss': losses.avg, 'accuracy': accuracy.avg}

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

def train_model(run_id=None):
    """Train the PCOS classification model."""
    if run_id is None:
        run_id = create_run_id()
    
    print(f"Starting training run: {run_id}")
    
    # Create directories for this run
    run_dir = os.path.join(config.OUTPUT_DIR, run_id)
    model_dir = os.path.join(run_dir, "models")
    logs_dir = os.path.join(run_dir, "logs")
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(config.RANDOM_SEED)
    
    # Save configuration
    save_config(run_dir)
    
    # Get data loaders
    train_loader, val_loader, test_loader, _ = get_dataloaders()
    
    # Create model
    model = PCOSClassifier()
    model = model.to(config.DEVICE)
    
    # Define loss function
    criterion = nn.BCELoss()
    
    # Initialize history dictionary
    history = {
        'phase1': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_precision': [], 'val_recall': []},
        'phase2': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_precision': [], 'val_recall': []}
    }
    
    # Define early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, verbose=True)
    
    # Phase 1: Train with frozen backbones
    print("\nPhase 1: Training with frozen backbones")
    
    # Define optimizer for phase 1
    optimizer = optim.Adam(model.parameters(), lr=config.INITIAL_LR)
    
    # Define learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config.LR_REDUCTION_FACTOR, 
        patience=config.LR_REDUCTION_PATIENCE, 
        verbose=True
    )
    
    # Training loop for phase 1
    for epoch in range(config.EPOCHS_PHASE1):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS_PHASE1}")
        
        # Train for one epoch
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, config.DEVICE)
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Update history
        history['phase1']['train_loss'].append(train_metrics['loss'])
        history['phase1']['train_acc'].append(train_metrics['accuracy'])
        history['phase1']['val_loss'].append(val_metrics['loss'])
        history['phase1']['val_acc'].append(val_metrics['accuracy'])
        history['phase1']['val_auc'].append(val_metrics['auc'])
        history['phase1']['val_precision'].append(val_metrics['precision'])
        history['phase1']['val_recall'].append(val_metrics['recall'])
        
        # Check early stopping
        early_stopping(val_metrics['loss'], model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    
    # Load the best model from phase 1
    model.load_state_dict(torch.load(os.path.join(run_dir, 'best_model.pt')))
    
    # Save phase 1 model
    torch.save(model.state_dict(), os.path.join(model_dir, 'phase1_model.pt'))
    
    # Phase 2: Fine-tuning with unfrozen backbones
    print("\nPhase 2: Fine-tuning with partially unfrozen backbones")
    
    # Unfreeze the last layers of the backbones
    model.unfreeze_backbones(unfreeze_percentage=0.3)

    # Define early stopping with the correct path
    best_model_path = os.path.join(run_dir, 'best_model.pt')
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE, 
        verbose=True,
        path=best_model_path
    )

    # Reset early stopping
    # early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, verbose=True)
    
    # Define optimizer for phase 2 with lower learning rate
    optimizer = optim.Adam(model.parameters(), lr=config.FINE_TUNING_LR)
    
    # Define learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config.LR_REDUCTION_FACTOR, 
        patience=config.LR_REDUCTION_PATIENCE, 
        verbose=True
    )
    
    # Training loop for phase 2
    for epoch in range(config.EPOCHS_PHASE2):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS_PHASE2}")
        
        # Train for one epoch
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, config.DEVICE)
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Update history
        history['phase2']['train_loss'].append(train_metrics['loss'])
        history['phase2']['train_acc'].append(train_metrics['accuracy'])
        history['phase2']['val_loss'].append(val_metrics['loss'])
        history['phase2']['val_acc'].append(val_metrics['accuracy'])
        history['phase2']['val_auc'].append(val_metrics['auc'])
        history['phase2']['val_precision'].append(val_metrics['precision'])
        history['phase2']['val_recall'].append(val_metrics['recall'])
        
        # Check early stopping
        early_stopping(val_metrics['loss'], model)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    
    # Load the best model from phase 2
    model.load_state_dict(torch.load(os.path.join(run_dir, 'best_model.pt')))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.pt'))
    
    # Save training history
    with open(os.path.join(run_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    return model, history, run_id

def train_with_cross_validation():
    """Train models using k-fold cross-validation."""
    if not config.PERFORM_CV:
        return train_model()
    
    # Create a run ID for this CV run
    cv_run_id = f"cv_{create_run_id()}"
    cv_dir = os.path.join(config.OUTPUT_DIR, cv_run_id)
    os.makedirs(cv_dir, exist_ok=True)
    
    # Get cross-validation folds
    folds = get_cross_validation_folds()
    
    # Train a model for each fold
    fold_results = []
    for fold_data in folds:
        print(f"\n=== Training Fold {fold_data['fold']}/{len(folds)} ===")
        
        fold_run_id = f"{cv_run_id}/fold_{fold_data['fold']}"
        
        # Create model
        model = PCOSClassifier()
        model = model.to(config.DEVICE)
        
        # Define loss function
        criterion = nn.BCELoss()
        
        # Initialize history dictionary
        history = {
            'phase1': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_precision': [], 'val_recall': []},
            'phase2': {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_precision': [], 'val_recall': []}
        }
        
        # Define early stopping

        # Create directories for this fold
        fold_dir = os.path.join(config.OUTPUT_DIR, fold_run_id)
        model_dir = os.path.join(fold_dir, "models")

        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # Define early stopping with the correct path
        best_model_path = os.path.join(fold_dir, 'best_model.pt')
        early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE, 
            verbose=True,
            path=best_model_path
        )

        # early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, verbose=True)
        
        # Create directories for this fold
        fold_dir = os.path.join(config.OUTPUT_DIR, fold_run_id)
        model_dir = os.path.join(fold_dir, "models")
        
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Phase 1: Train with frozen backbones
        print(f"\nFold {fold_data['fold']} - Phase 1: Training with frozen backbones")
        
        # Define optimizer for phase 1
        optimizer = optim.Adam(model.parameters(), lr=config.INITIAL_LR)
        
        # Define learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=config.LR_REDUCTION_FACTOR, 
            patience=config.LR_REDUCTION_PATIENCE, 
            verbose=True
        )
        
        # Training loop for phase 1
        for epoch in range(config.EPOCHS_PHASE1):
            print(f"\nEpoch {epoch+1}/{config.EPOCHS_PHASE1}")
            
            # Train for one epoch
            train_metrics = train_epoch(model, fold_data['train_loader'], criterion, optimizer, config.DEVICE)
            
            # Validate
            val_metrics = validate(model, fold_data['val_loader'], criterion, config.DEVICE)
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
            
            # Update learning rate
            scheduler.step(val_metrics['loss'])
            
            # Update history
            history['phase1']['train_loss'].append(train_metrics['loss'])
            history['phase1']['train_acc'].append(train_metrics['accuracy'])
            history['phase1']['val_loss'].append(val_metrics['loss'])
            history['phase1']['val_acc'].append(val_metrics['accuracy'])
            history['phase1']['val_auc'].append(val_metrics['auc'])
            history['phase1']['val_precision'].append(val_metrics['precision'])
            history['phase1']['val_recall'].append(val_metrics['recall'])
            
            # Check early stopping
            early_stopping(val_metrics['loss'], model)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
        
        # Save phase 1 model
        torch.save(model.state_dict(), os.path.join(model_dir, 'phase1_model.pt'))
        
        # Load the best model from phase 1
        model.load_state_dict(torch.load(os.path.join(fold_dir, 'best_model.pt')))
        
        # Phase 2: Fine-tuning with unfrozen backbones
        print(f"\nFold {fold_data['fold']} - Phase 2: Fine-tuning with partially unfrozen backbones")
        
        # Unfreeze the last layers of the backbones
        model.unfreeze_backbones(unfreeze_percentage=0.3)
        
        # Reset early stopping
        early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, verbose=True)
        
        # Define optimizer for phase 2 with lower learning rate
        optimizer = optim.Adam(model.parameters(), lr=config.FINE_TUNING_LR)
        
        # Define learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=config.LR_REDUCTION_FACTOR, 
            patience=config.LR_REDUCTION_PATIENCE, 
            verbose=True
        )
        
        # Training loop for phase 2
        for epoch in range(config.EPOCHS_PHASE2):
            print(f"\nEpoch {epoch+1}/{config.EPOCHS_PHASE2}")
            
            # Train for one epoch
            train_metrics = train_epoch(model, fold_data['train_loader'], criterion, optimizer, config.DEVICE)
            
            # Validate
            val_metrics = validate(model, fold_data['val_loader'], criterion, config.DEVICE)
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
            
            # Update learning rate
            scheduler.step(val_metrics['loss'])
            
            # Update history
            history['phase2']['train_loss'].append(train_metrics['loss'])
            history['phase2']['train_acc'].append(train_metrics['accuracy'])
            history['phase2']['val_loss'].append(val_metrics['loss'])
            history['phase2']['val_acc'].append(val_metrics['accuracy'])
            history['phase2']['val_auc'].append(val_metrics['auc'])
            history['phase2']['val_precision'].append(val_metrics['precision'])
            history['phase2']['val_recall'].append(val_metrics['recall'])
            
            # Check early stopping
            early_stopping(val_metrics['loss'], model)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
        
        # Save final model for this fold
        torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.pt'))
        
        # Load the best model from phase 2
        model.load_state_dict(torch.load(os.path.join(fold_dir, 'best_model.pt')))
        
        # Save training history for this fold
        with open(os.path.join(fold_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=4)
        
        # Final validation of the best model
        final_metrics = validate(model, fold_data['val_loader'], criterion, config.DEVICE)
        
        # Save fold results
        fold_results.append({
            'fold': fold_data['fold'],
            'history': history,
            'metrics': final_metrics
        })
    
    # Save CV results summary
    with open(os.path.join(cv_dir, 'cv_results.json'), 'w') as f:
        # Convert metrics to serializable format
        serializable_results = []
        for result in fold_results:
            serializable_result = {
                'fold': result['fold'],
                'history': result['history'],
                'metrics': {k: float(v) for k, v in result['metrics'].items()}
            }
            serializable_results.append(serializable_result)
        
        json.dump(serializable_results, f, indent=4)
    
    # Find the best fold based on validation AUC
    best_fold_idx = np.argmax([result['metrics']['auc'] for result in fold_results])
    best_fold = fold_results[best_fold_idx]['fold']
    best_fold_dir = os.path.join(config.OUTPUT_DIR, f"{cv_run_id}/fold_{best_fold}")
    
    # Load the best model
    best_model = PCOSClassifier()
    best_model.load_state_dict(torch.load(os.path.join(best_fold_dir, 'best_model.pt')))
    best_model = best_model.to(config.DEVICE)
    
    return best_model, fold_results, cv_run_id

if __name__ == "__main__":
    # Train model
    if config.PERFORM_CV:
        model, results, run_id = train_with_cross_validation()
    else:
        model, results, run_id = train_model()
    
    print(f"Training completed. Results saved to {os.path.join(config.OUTPUT_DIR, run_id)}")