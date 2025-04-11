# run_inference.py
import os
import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
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

def run_inference(model_path, image_path, output_dir=None, show_gradcam=True):
    """Run inference on a single image."""
    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = PCOSClassifier()
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    model.eval()
    
    # Process image
    img_tensor = preprocess_image(image_path)
    
    # Get prediction
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(config.DEVICE))
        prob = output.item()
        prediction = "PCOS" if prob > 0.5 else "Normal"
    
    # Unnormalize image for display
    inv_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    ])
    
    img_unnorm = inv_normalize(img_tensor).cpu().numpy().transpose(1, 2, 0)
    img_unnorm = np.clip(img_unnorm, 0, 1)
    
    # Create figure
    fig = plt.figure(figsize=(12, 6))
    
    if show_gradcam:
        # Generate GradCAM
        plt.subplot(1, 2, 1)
        plt.imshow(img_unnorm)
        plt.title(f'Prediction: {prediction} ({prob:.4f})')
        plt.axis('off')
        
        # Try different backbones for GradCAM
        for i, layer_name in enumerate([
            "feature_extractors.2.features.denseblock4", 
            "feature_extractors.0.features.28"
        ]):
            try:
                plt.subplot(1, 2, 2)
                grad_cam = generate_gradcam(
                    model, 
                    img_tensor.to(config.DEVICE),
                    layer_name,
                    save_path=None  # Don't save separately
                )
                break
            except:
                if i == 1:  # Last attempt failed
                    plt.subplot(1, 2, 2)
                    plt.imshow(img_unnorm)
                    plt.title("GradCAM not available")
                    plt.axis('off')
    else:
        plt.imshow(img_unnorm)
        plt.title(f'Prediction: {prediction} ({prob:.4f})')
        plt.axis('off')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'prediction_{os.path.basename(image_path)}'), 
                   dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
    
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Prediction: {prediction} (Probability: {prob:.4f})")
    
    return {"image": image_path, "prediction": prediction, "probability": prob}

def main():
    parser = argparse.ArgumentParser(description='Run inference on new images')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--dir', type=str, help='Path to directory with images')
    parser.add_argument('--output', type=str, default='inference_results', help='Output directory')
    parser.add_argument('--no-gradcam', action='store_true', help='Disable GradCAM visualization')
    
    args = parser.parse_args()
    
    if not args.image and not args.dir:
        parser.error("Either --image or --dir must be provided")
    
    if args.image:
        run_inference(args.model, args.image, args.output, not args.no_gradcam)
    
    if args.dir:
        results = []
        for file_name in os.listdir(args.dir):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(args.dir, file_name)
                result = run_inference(args.model, image_path, args.output, not args.no_gradcam)
                results.append(result)
        
        # Save summary
        import json
        with open(os.path.join(args.output, 'inference_summary.json'), 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()