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
        """
        self.data_dir = data_dir
        self.df = df.reset_index(drop=True)
        self.image_dir = os.path.join(data_dir, image_dir)
        self.mask_dir = os.path.join(data_dir, mask_dir)
        self.transform = transform
        self.is_test = is_test
        self.img_size = img_size
        self.patch_size = patch_size
        self.scale = scale
        
        # Pre-compute valid patch coordinates for each WSI
        self.patches = self._precompute_patches()
        
    def _precompute_patches(self) -> List[Dict]:
        """Pre-compute valid patch coordinates for all WSIs."""
        patches = []
        for idx in range(len(self.df)):
            img_id = self.df.iloc[idx]['image_id']
            img_path = os.path.join(self.image_dir, f"{img_id}.tiff")
            
            with openslide.OpenSlide(img_path) as slide:
                w, h = slide.dimensions
                
            # Calculate number of patches in each dimension
            n_w = (w // self.img_size) - 1
            n_h = (h // self.img_size) - 1
            
            # Store patch coordinates
            for i in range(n_w):
                for j in range(n_h):
                    x = i * self.img_size
                    y = j * self.img_size
                    patches.append({
                        'idx': idx,
                        'x': x,
                        'y': y,
                        'img_id': img_id
                    })
        return patches
    
    def __len__(self) -> int:
        return len(self.patches)
    
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
                mask_region = (mask_region[..., 0] > 0).astype(np.float32)
            else:
                mask_region = np.zeros((size, size), dtype=np.float32)
                
        return mask_region
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset."""
        patch_info = self.patches[idx]
        img_id = patch_info['img_id']
        x, y = patch_info['x'], patch_info['y']
        
        # Load image
        img_path = os.path.join(self.image_dir, f"{img_id}.tiff")
        with openslide.OpenSlide(img_path) as slide:
            image = self._load_wsi_region(slide, x, y, self.img_size)
        
        # Load mask if not in test mode
        mask = None
        if not self.is_test:
            mask_path = os.path.join(self.mask_dir, f"{img_id}_mask.tiff")
            mask = self._load_mask_region(mask_path, x, y, self.img_size)
        
        # Get label from dataframe
        label = self.df.loc[self.df['image_id'] == img_id, 'isup_grade'].values[0]
        
        # Apply transforms
        if self.transform:
            if mask is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            else:
                transformed = self.transform(image=image)
                image = transformed['image']
        
        # Prepare output
        sample = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'image_id': img_id,
            'x': x,
            'y': y
        }
        
        if mask is not None:
            sample['mask'] = mask
            
        return sample
