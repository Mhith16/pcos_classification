"""Configuration settings for the PCOS classification project."""

import os
import torch

# Paths
DATA_DIR = "PCOS"
OUTPUT_DIR = "output"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Create directories if they don't exist
for directory in [OUTPUT_DIR, MODEL_DIR, LOGS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data settings
IMG_SIZE = (299, 299)
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Model settings
BACKBONE_MODELS = ["vgg16", "inception_v3", "densenet201"]
USE_PRETRAINED = True
FREEZE_BACKBONE = True

# Vision Transformer settings
TRANSFORMER_LAYERS = 4
NUM_HEADS = 8
EMBEDDING_DIM = 512
MLP_DIM = 1024
DROPOUT_RATE = 0.1

# Training settings
INITIAL_LR = 1e-4
FINE_TUNING_LR = 1e-5
EPOCHS_PHASE1 = 20  # Training with frozen backbone
EPOCHS_PHASE2 = 30  # Fine-tuning
EARLY_STOPPING_PATIENCE = 10
LR_REDUCTION_PATIENCE = 5
LR_REDUCTION_FACTOR = 0.1

# Cross-validation
PERFORM_CV = True
NUM_FOLDS = 5

# Seed for reproducibility
RANDOM_SEED = 42