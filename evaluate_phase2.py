# evaluate_phase2.py
import os
import torch
import config
from models import PCOSClassifier
from dataset import get_dataloaders
from evaluate import evaluate_model, evaluate_training_history
import json
import argparse

def evaluate_phase2_model(run_id, fold=1):
    # Load model paths
    fold_dir = os.path.join(config.OUTPUT_DIR, run_id, f"fold_{fold}")
    model_dir = os.path.join(fold_dir, "models")
    final_model_path = os.path.join(model_dir, "final_model.pt")
    
    # Create evaluation directory
    eval_dir = os.path.join(fold_dir, "evaluation_phase2")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Load model
    model = PCOSClassifier()
    model.load_state_dict(torch.load(final_model_path, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    
    # Get test data
    _, _, test_loader, _ = get_dataloaders()
    
    # Run evaluation
    print(f"Evaluating Phase 2 model from {final_model_path}")
    metrics = evaluate_model(model, test_loader, os.path.join(run_id, f"fold_{fold}/phase2"))
    
    # Load training history
    if os.path.exists(os.path.join(fold_dir, "phase2_history.json")):
        with open(os.path.join(fold_dir, "phase2_history.json"), 'r') as f:
            history = json.load(f)
        evaluate_training_history({"phase2": history}, os.path.join(run_id, f"fold_{fold}/phase2"))
    
    # Save results
    with open(os.path.join(eval_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\nEvaluation Results:")
    for key, value in metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Phase 2 model')
    parser.add_argument('--run_id', type=str, required=True, help='Run ID to evaluate')
    parser.add_argument('--fold', type=int, default=1, help='Fold number')
    
    args = parser.parse_args()
    evaluate_phase2_model(args.run_id, args.fold)