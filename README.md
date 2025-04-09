# PCOS Classification Using Multiple Pretrained Models with Vision Transformer

This project implements a deep learning approach for PCOS (Polycystic Ovary Syndrome) classification from ultrasound images. The model architecture combines features from three pretrained models (VGG16, Inception-V3, and DenseNet201) and feeds them into a Vision Transformer for the final classification.

## Features

- Multi-model ensemble approach with three pretrained backbones
- Vision Transformer for feature integration and classification
- Comprehensive evaluation metrics and visualizations
- Cross-validation support for robust evaluation
- GradCAM visualization for model interpretability
- Implemented with PyTorch for research flexibility

## Project Structure

- `config.py`: Configuration settings for the project
- `dataset.py`: Data loading and preprocessing functions
- `models.py`: Model architecture definition
- `train.py`: Training script with two-phase training strategy
- `evaluate.py`: Comprehensive evaluation with metrics and visualizations
- `visualization.py`: Visualization functions
- `utils.py`: Utility functions
- `inference.py`: Inference script for making predictions

## Usage

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd pcos-classification

# Install dependencies
pip install -r requirements.txt