import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import openslide
from typing import Dict, List, Tuple, Optional, Union

class PANDA_Dataset(Dataset):
    """
    PANDA Challenge Dataset for prostate cancer detection.
    Handles loading of WSI patches and corresponding masks.
    """
    
    def __init__(
        self,
        data_dir: str,
        df: pd.DataFrame,
        image_dir: str = 'train_images',
        mask_dir: str = 'train_label_masks',
        transform=None,
        is_test: bool = False,
        img_size: int = 256,
        patch_size: int = 224,
        scale: int = 1,
        patches_per_image: int = 16,  # Number of random patches per image
    ):
        """
        Args:
            data_dir: Root directory of the dataset
            df: DataFrame containing image metadata and labels
            image_dir: Name of the directory containing WSI images
            mask_dir: Name of the directory containing mask images
            transform: Optional transform to be applied on a sample
            is_test: If True, skips loading masks
            img_size: Size of the extracted patches
            patch_size: Final size of the patches after transforms
            scale: Scale factor for the WSI
            patches_per_image: Number of patches to extract per image
        """
        self.data_dir = data_dir
        self.df = df.reset_index(drop=True)
        
        # Clean labels: map 'negative' to '0+0' in gleason_score
        if 'gleason_score' in self.df.columns:
            self.df['gleason_score'] = self.df['gleason_score'].apply(lambda x: '0+0' if x == 'negative' else x)
            # Drop known mislabels (from Kaggle EDA)
            # Check if row 7273 exists and drop it
            if 7273 in self.df.index:
                self.df = self.df.drop([7273])
        
        self.image_dir = os.path.join(data_dir, image_dir)
        self.mask_dir = os.path.join(data_dir, mask_dir)
        self.transform = transform
        self.is_test = is_test
        self.img_size = img_size
        self.patch_size = patch_size
        self.scale = scale
        self.patches_per_image = patches_per_image
        
        # Cache for WSI dimensions (loaded lazily)
        self._dimensions_cache = {}
        
    def _get_wsi_dimensions(self, img_path: str) -> Tuple[int, int]:
        """Get WSI dimensions with caching."""
        if img_path in self._dimensions_cache:
            return self._dimensions_cache[img_path]
            
        try:
            with openslide.OpenSlide(img_path) as slide:
                w, h = slide.dimensions
            
            self._dimensions_cache[img_path] = (w, h)
            return w, h
            
        except Exception as e:
            print(f"[WARNING] Could not get dimensions for {img_path}: {e}")
            # Return default dimensions
            return 1024, 1024
    
    def _get_random_patch_coordinates(self, img_id: str) -> Tuple[int, int]:
        """Get random patch coordinates for a given image."""
        img_path = os.path.join(self.image_dir, f"{img_id}.tiff")
        w, h = self._get_wsi_dimensions(img_path)
        
        # Calculate valid coordinate ranges
        max_x = max(0, w - self.img_size)
        max_y = max(0, h - self.img_size)
        
        # Generate random coordinates
        x = np.random.randint(0, max_x + 1) if max_x > 0 else 0
        y = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        
        return x, y
    
    def __len__(self) -> int:
        return len(self.df) * self.patches_per_image
    
    def _load_wsi_region(
        self,
        slide: openslide.OpenSlide,
        x: int,
        y: int,
        size: int
    ) -> np.ndarray:
        """Load a region from a WSI."""
        region = slide.read_region((x, y), 0, (size, size))
        region = region.convert('RGB')
        return np.array(region)
    
    def _load_mask_region(
        self,
        mask_path: str,
        x: int,
        y: int,
        size: int
    ) -> Optional[np.ndarray]:
        """Load a region from a mask WSI."""
        if not os.path.exists(mask_path):
            return None
            
        with openslide.OpenSlide(mask_path) as mask_slide:
            mask_region = mask_slide.read_region((x, y), 0, (size, size))
            mask_region = mask_region.convert('RGB')
            mask_region = np.array(mask_region)
            
            # Convert RGB mask to binary mask (assuming red channel indicates tumor)
            if mask_region.size > 0:
                mask_region = (mask_region[..., 0] > 0).astype('float32')
            else:
                mask_region = np.zeros((size, size), dtype='float32')
                
        return mask_region
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset."""
        # Calculate which image and which patch within that image
        img_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image
        
        # Get image info
        img_id = str(self.df.iloc[img_idx]['image_id'])
        
        # Get random patch coordinates
        x, y = self._get_random_patch_coordinates(img_id)
        
        # Load image
        img_path = os.path.join(self.image_dir, f"{img_id}.tiff")
        try:
            with openslide.OpenSlide(img_path) as slide:
                image = self._load_wsi_region(slide, x, y, self.img_size)
        except Exception as e:
            print(f"[WARNING] Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = np.zeros((self.img_size, self.img_size, 3), dtype='uint8')
        
        # Load mask if not in test mode
        mask = None
        if not self.is_test:
            mask_path = os.path.join(self.mask_dir, f"{img_id}_mask.tiff")
            mask = self._load_mask_region(mask_path, x, y, self.img_size)
            
        # If no mask is loaded, create a dummy mask of zeros
        if mask is None:
            mask = np.zeros((self.img_size, self.img_size), dtype='float32')
        
        # Get label from dataframe
        label = self.df.iloc[img_idx]['isup_grade']
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Prepare output
        sample = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'image_id': img_id,
            'x': x,
            'y': y,
            'mask': mask  # Always include mask, even if it's dummy
        }
        
        return sample
