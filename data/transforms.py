import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Dict, Any, Optional

def get_train_transforms(img_size: int = 224, scale: float = 0.1) -> A.Compose:
    """
    Get training data augmentation transforms.
    
    Args:
        img_size: Size of the output image
        scale: Scale factor for random scaling
        
    Returns:
        A.Compose: Composed transforms
    """
    return A.Compose([
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomScale(scale_limit=scale, p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,  # 1/16 of image size
            scale_limit=0.1,
            rotate_limit=10,
            p=0.5,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        
        # Color transforms
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=10,
                val_shift_limit=10,
                p=1.0
            ),
            A.RGBShift(
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0
            ),
        ], p=0.5),
        
        # Spatial preserving transforms
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(),
            A.MotionBlur(),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.GlassBlur(sigma=0.7, max_delta=2, p=1.0),
        ], p=0.5),
        
        # Normalization and resizing
        A.Resize(img_size, img_size, always_apply=True),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        
        # Convert to PyTorch tensor
        ToTensorV2(transpose_mask=True),
    ], additional_targets={'mask': 'mask'})

def get_valid_transforms(img_size: int = 224) -> A.Compose:
    """
    Get validation/test data transforms (only resizing and normalization).
    
    Args:
        img_size: Size of the output image
        
    Returns:
        A.Compose: Composed transforms
    """
    return A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2(transpose_mask=True),
    ], additional_targets={'mask': 'mask'})
