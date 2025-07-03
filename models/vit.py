import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from transformers import ViTModel, ViTConfig

class ViTForProstateCancer(nn.Module):
    """
    Vision Transformer model for prostate cancer detection with multi-task learning.
    Performs both classification (ISUP grade) and segmentation (tumor region) tasks.
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_classes: int = 6,  # ISUP grades 0-5
        pretrained: bool = True,
        img_size: int = 224,
        patch_size: int = 16,
        **kwargs
    ) -> None:
        """
        Initialize the model.
        
        Args:
            model_name: Name of the Vision Transformer model
            num_classes: Number of ISUP grade classes
            pretrained: Whether to use pre-trained weights
            img_size: Input image size
            patch_size: Size of patches
        """
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Initialize the base ViT model
        try:
            if pretrained:
                self.vit = ViTModel.from_pretrained(model_name, add_pooling_layer=False)
            else:
                config = ViTConfig(
                    image_size=img_size,
                    patch_size=patch_size,
                    num_channels=3,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    intermediate_size=3072,
                    hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1,
                    initializer_range=0.02,
                    layer_norm_eps=1e-12,
                    is_encoder_decoder=False,
                )
                self.vit = ViTModel(config, add_pooling_layer=False)
        except Exception as e:
            print(f"[WARNING] Failed to load pretrained model {model_name}: {e}")
            print("[INFO] Falling back to randomly initialized ViT model")
            config = ViTConfig(
                image_size=img_size,
                patch_size=patch_size,
                num_channels=3,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
                is_encoder_decoder=False,
            )
            self.vit = ViTModel(config, add_pooling_layer=False)
        
        # Classification head for ISUP grade prediction
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        # Segmentation head for tumor region prediction
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(
                in_channels=768,
                out_channels=256,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights for classification and segmentation heads."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        for m in self.segmentation_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass of the model.
        
        Args:
            pixel_values: Input images [batch_size, 3, H, W]
            labels: ISUP grade labels [batch_size, ]
            masks: Segmentation masks [batch_size, 1, H, W]
            
        Returns:
            Dict containing classification logits and segmentation logits
        """
        batch_size = pixel_values.shape[0]
        
        # Get features from ViT
        outputs = self.vit(pixel_values)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Classification: use [CLS] token
        cls_token = sequence_output[:, 0]  # [batch_size, hidden_size]
        logits = self.classifier(cls_token)  # [batch_size, num_classes]
        
        # Segmentation: reshape patch embeddings to 2D feature maps
        patch_embeddings = sequence_output[:, 1:]  # Remove [CLS] token
        h = w = int(patch_embeddings.shape[1] ** 0.5)
        patch_embeddings = patch_embeddings.permute(0, 2, 1).view(
            batch_size, -1, h, w
        )  # [batch_size, hidden_size, h, w]
        
        # Upsample to input resolution
        seg_logits = self.segmentation_head(patch_embeddings)
        
        # Resize segmentation logits to match mask size if masks are provided
        if masks is not None:
            target_size = masks.shape[-2:]  # Get H, W from masks
            seg_logits = F.interpolate(
                seg_logits, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        # Calculate losses if labels are provided
        loss = None
        if labels is not None:
            # Classification loss
            cls_loss = F.cross_entropy(logits, labels)
            
            # Segmentation loss (only if masks are provided)
            if masks is not None:
                seg_loss = self.dice_bce_loss(seg_logits, masks)
                # Combined loss with weight
                alpha = 0.7  # Weight for classification loss
                loss = alpha * cls_loss + (1 - alpha) * seg_loss
            else:
                # Only classification loss if no masks
                loss = cls_loss
        
        return {
            'logits': logits,
            'seg_logits': seg_logits,
            'loss': loss
        }
    
    @staticmethod
    def dice_bce_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1.0
    ) -> torch.Tensor:
        """
        Dice loss + Binary Cross Entropy loss for segmentation.
        
        Args:
            pred: Predicted probabilities [batch_size, 1, H, W]
            target: Ground truth masks [batch_size, 1, H, W]
            smooth: Smoothing factor
            
        Returns:
            Combined Dice + BCE loss
        """
        # Flatten predictions and targets
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy(pred_flat, target_flat)
        
        # Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        # Combined loss
        return bce + (1 - dice)
