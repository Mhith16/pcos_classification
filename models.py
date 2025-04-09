"""Model architecture for PCOS classification using PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, inception_v3, densenet201, VGG16_Weights, Inception_V3_Weights, DenseNet201_Weights
import config
from einops import rearrange

class FeatureExtractor(nn.Module):
    """Feature extractor module using pretrained models."""
    
    def __init__(self, backbone_name, trainable=False):
        super(FeatureExtractor, self).__init__()
        
        self.backbone_name = backbone_name
        
        if backbone_name == "vgg16":
            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            self.features = model.features
            self.avgpool = model.avgpool
            self.flatten = nn.Flatten()
            self.classifier = nn.Sequential(*list(model.classifier.children())[:4])  # Up to relu after fc1
            self.feature_dim = 4096
            
        elif backbone_name == "inception_v3":
            model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
            # Disable the auxiliary classifier to avoid size issues
            model.aux_logits = False
            model.AuxLogits = None
            # Remove the final classifier
            model.fc = nn.Identity()
            self.features = model
            self.feature_dim = 2048
            
        elif backbone_name == "densenet201":
            model = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
            # Remove the final classifier
            model.classifier = nn.Identity()
            self.features = model
            self.feature_dim = 1920
            
        else:
            raise ValueError(f"Unsupported backbone model: {backbone_name}")
        
        # Freeze model parameters if not trainable
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        if self.backbone_name == "vgg16":
            x = self.features(x)
            x = self.avgpool(x)
            x = self.flatten(x)
            x = self.classifier(x)
        else:
            x = self.features(x)
            # Apply global average pooling if not VGG
            if x.dim() > 2:
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)
        
        return x

class TransformerEncoder(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        # Multi-head attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Multi-head attention
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_output)
        
        # MLP
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output
        
        return x

class PCOSClassifier(nn.Module):
    """Full model with multiple backbones and Vision Transformer."""
    
    def __init__(self):
        super(PCOSClassifier, self).__init__()
        
        # Create feature extractors
        self.feature_extractors = nn.ModuleList([
            FeatureExtractor(backbone, trainable=not config.FREEZE_BACKBONE) 
            for backbone in config.BACKBONE_MODELS
        ])
        
        # Calculate total feature dimension
        total_feature_dim = sum(extractor.feature_dim for extractor in self.feature_extractors)
        
        # Projection to embedding dimension
        self.projection = nn.Linear(total_feature_dim, config.EMBEDDING_DIM)
        self.norm = nn.LayerNorm(config.EMBEDDING_DIM)
        
        # Transformer encoders
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(
                config.EMBEDDING_DIM, 
                config.NUM_HEADS, 
                config.MLP_DIM, 
                config.DROPOUT_RATE
            ) for _ in range(config.TRANSFORMER_LAYERS)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(config.EMBEDDING_DIM)
        
        # Classification head
        self.classifier = nn.Linear(config.EMBEDDING_DIM, 1)
    
    def forward(self, x):
        # Extract features from each backbone
        features = []
        for extractor in self.feature_extractors:
            features.append(extractor(x))
        
        # Concatenate features
        if len(features) > 1:
            concatenated_features = torch.cat(features, dim=1)
        else:
            concatenated_features = features[0]
        
        # Project to embedding dimension
        x = self.projection(concatenated_features)
        x = self.norm(x)
        
        # Add batch dimension if necessary for transformer (B x 1 x D)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Final layer normalization
        x = self.final_norm(x)
        
        # Squeeze sequence dimension for classification
        if x.dim() == 3:
            x = x.squeeze(1)
        
        # Apply classification head
        x = self.classifier(x)
        
        return torch.sigmoid(x)
    
    def unfreeze_backbones(self, unfreeze_percentage=0.3):
        """Unfreeze the last few layers of each backbone for fine-tuning."""
        for extractor in self.feature_extractors:
            # Get all parameters
            all_params = list(extractor.parameters())
            num_params = len(all_params)
            
            # Calculate how many parameters to unfreeze
            num_to_unfreeze = int(num_params * unfreeze_percentage)
            
            # Unfreeze the last X% of parameters
            for param in all_params[-num_to_unfreeze:]:
                param.requires_grad = True
            
            # Always unfreeze BatchNorm layers
            for module in extractor.modules():
                if isinstance(module, nn.BatchNorm2d):
                    for param in module.parameters():
                        param.requires_grad = True