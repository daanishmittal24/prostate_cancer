#!/usr/bin/env python3
"""
TransUNet Implementation for Prostate Cancer Detection
Supports both classification (ISUP grading) and segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # Project and flatten: [B, embed_dim, H/P, W/P] -> [B, embed_dim, N]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self Attention"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """Feed Forward Network"""
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block with LayerNorm, Attention, and MLP"""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CNNBlock(nn.Module):
    """Convolutional block for decoder"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UpBlock(nn.Module):
    """Upsampling block for decoder"""
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = CNNBlock(in_channels, out_channels)
        
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class TransUNet(nn.Module):
    """
    TransUNet for Prostate Cancer Detection
    Combines Vision Transformer with U-Net decoder for classification + segmentation
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 6,
        seg_classes: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pretrained: bool = False,
        pretrained_path: Optional[str] = None
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        
        # Classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Segmentation decoder
        self.decoder = self._build_decoder(embed_dim, seg_classes)
        
        # Initialize weights
        self._init_weights()
        
        # Load pre-trained weights if specified
        if pretrained:
            self._load_pretrained_weights(pretrained_path)
    
    def _build_decoder(self, embed_dim: int, seg_classes: int):
        """Build U-Net style decoder"""
        # Calculate feature map size after patch embedding
        feat_size = self.img_size // self.patch_size  # 14 for 224x224 with 16x16 patches
        
        # Project transformer features to convolutional features
        self.feat_proj = nn.Linear(embed_dim, embed_dim)
        
        # Decoder blocks
        decoder = nn.ModuleList([
            # Upsample from 14x14 to 28x28
            UpBlock(embed_dim, 512),
            CNNBlock(512, 512),
            
            # Upsample from 28x28 to 56x56  
            UpBlock(512, 256),
            CNNBlock(256, 256),
            
            # Upsample from 56x56 to 112x112
            UpBlock(256, 128),
            CNNBlock(128, 128),
            
            # Upsample from 112x112 to 224x224
            UpBlock(128, 64),
            CNNBlock(64, 64),
            
            # Final segmentation head
            nn.Conv2d(64, seg_classes, kernel_size=1)
        ])
        
        return decoder
    
    def _init_weights(self):
        """Initialize weights"""
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize other weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _load_pretrained_weights(self, pretrained_path: Optional[str] = None):
        """Load pre-trained weights from timm or custom path"""
        try:
            import timm
            print("Loading pre-trained ViT weights from timm...")
            
            # Load pre-trained ViT model
            pretrained_model = timm.create_model('vit_base_patch16_224', pretrained=True)
            pretrained_dict = pretrained_model.state_dict()
            
            # Get current model state dict
            model_dict = self.state_dict()
            
            # Filter out unnecessary keys and match dimensions
            filtered_dict = {}
            for k, v in pretrained_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                    print(f"Loaded pre-trained weight: {k}")
                elif k.startswith('head.'):
                    # Skip classification head (different number of classes)
                    continue
                else:
                    print(f"Skipped pre-trained weight: {k} (shape mismatch or not found)")
            
            # Update model dict
            model_dict.update(filtered_dict)
            self.load_state_dict(model_dict)
            print(f"Successfully loaded {len(filtered_dict)} pre-trained weights")
            
        except ImportError:
            print("timm not installed, trying to load from torchvision...")
            try:
                import torchvision.models as models
                
                # Load pre-trained ResNet as backbone for CNN features
                resnet = models.resnet50(pretrained=True)
                
                # Extract conv1 weights for patch embedding
                conv1_weight = resnet.conv1.weight.data
                
                # Adapt to patch embedding
                if self.patch_embed.proj.weight.shape == conv1_weight.shape:
                    self.patch_embed.proj.weight.data = conv1_weight
                    print("Loaded ResNet conv1 weights for patch embedding")
                else:
                    print("ResNet conv1 weights don't match patch embedding dimensions")
                    
            except Exception as e:
                print(f"Failed to load pre-trained weights: {e}")
                print("Continuing with random initialization...")
                
        except Exception as e:
            print(f"Failed to load pre-trained weights: {e}")
            print("Continuing with random initialization...")
    
    def forward_features(self, x):
        """Forward through transformer encoder"""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, embed_dim]
        
        # Add classification token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, embed_dim]
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x
    
    def forward_classification(self, features):
        """Forward through classification head"""
        # Use [CLS] token for classification
        cls_features = features[:, 0]  # [B, embed_dim]
        logits = self.classification_head(cls_features)
        return logits
    
    def forward_segmentation(self, features):
        """Forward through segmentation decoder"""
        B = features.shape[0]
        
        # Remove [CLS] token and reshape to feature map
        patch_features = features[:, 1:]  # [B, N, embed_dim]
        patch_features = self.feat_proj(patch_features)  # [B, N, embed_dim]
        
        # Reshape to 2D feature map
        feat_size = int(math.sqrt(patch_features.shape[1]))  # 14
        patch_features = patch_features.transpose(1, 2).reshape(B, self.embed_dim, feat_size, feat_size)
        
        # Apply decoder
        x = patch_features
        for layer in self.decoder:
            x = layer(x)
        
        return x
    
    def forward(self, x):
        """Forward pass returning both classification and segmentation outputs"""
        # Shared feature extraction
        features = self.forward_features(x)
        
        # Classification output
        classification_logits = self.forward_classification(features)
        
        # Segmentation output
        segmentation_logits = self.forward_segmentation(features)
        
        return {
            'classification': classification_logits,
            'segmentation': segmentation_logits
        }


def create_transunet(
    img_size: int = 224,
    num_classes: int = 6,
    seg_classes: int = 1,
    **kwargs
) -> TransUNet:
    """
    Create TransUNet model with default configurations
    
    Args:
        img_size: Input image size
        num_classes: Number of classification classes (ISUP grades)
        seg_classes: Number of segmentation classes
        **kwargs: Additional model parameters
    
    Returns:
        TransUNet model
    """
    # Set default values for parameters not explicitly passed
    default_params = {
        'patch_size': 16,
        'in_chans': 3,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
    }
    
    # Update defaults with any provided kwargs
    for key, value in default_params.items():
        if key not in kwargs:
            kwargs[key] = value
    
    model = TransUNet(
        img_size=img_size,
        num_classes=num_classes,
        seg_classes=seg_classes,
        **kwargs
    )
    return model


# Model size variants
def create_transunet_small(**kwargs) -> TransUNet:
    """Create small TransUNet model"""
    # Set defaults for small model
    small_defaults = {
        'img_size': 224,
        'patch_size': 16,
        'in_chans': 3,
        'num_classes': 6,
        'seg_classes': 1,
        'embed_dim': 384,
        'depth': 6,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
    }
    
    # Update defaults with any provided kwargs
    for key, value in small_defaults.items():
        if key not in kwargs:
            kwargs[key] = value
    
    return TransUNet(**kwargs)


def create_transunet_base(**kwargs) -> TransUNet:
    """Create base TransUNet model (default)"""
    return create_transunet(**kwargs)


def create_transunet_large(**kwargs) -> TransUNet:
    """Create large TransUNet model"""
    # Set defaults for large model
    large_defaults = {
        'img_size': 224,
        'patch_size': 16,
        'in_chans': 3,
        'num_classes': 6,
        'seg_classes': 1,
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
    }
    
    # Update defaults with any provided kwargs
    for key, value in large_defaults.items():
        if key not in kwargs:
            kwargs[key] = value
    
    return TransUNet(**kwargs)


if __name__ == "__main__":
    # Test the model
    model = create_transunet(img_size=224, num_classes=6, seg_classes=1)
    x = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(x)
        print(f"Classification output shape: {outputs['classification'].shape}")
        print(f"Segmentation output shape: {outputs['segmentation'].shape}")
        
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
