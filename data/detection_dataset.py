import os
import cv2
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import openslide
from typing import Dict, List, Tuple, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
from detectron2.structures import BoxMode
import pycocotools.mask as mask_util

class ProstateDetectionDataset(Dataset):
    """
    Prostate Cancer Detection Dataset for ViTDet + Mask R-CNN.
    Converts WSI patches to COCO format for object detection.
    """
    
    def __init__(
        self,
        data_dir: str,
        df: pd.DataFrame,
        image_dir: str = 'train_images',
        mask_dir: str = 'train_label_masks',
        transform=None,
        is_test: bool = False,
        patch_size: int = 1024,
        overlap: int = 128,
        min_tumor_area: int = 1000,
    ):
        """
        Args:
            data_dir: Root directory of the dataset
            df: DataFrame containing image metadata and labels
            image_dir: Name of the directory containing WSI images
            mask_dir: Name of the directory containing mask images
            transform: Albumentations transform pipeline
            is_test: If True, skips loading masks
            patch_size: Size of extracted patches
            overlap: Overlap between patches
            min_tumor_area: Minimum tumor area to include
        """
        self.data_dir = data_dir
        self.df = df.reset_index(drop=True)
        
        # Clean labels
        if 'gleason_score' in self.df.columns:
            self.df['gleason_score'] = self.df['gleason_score'].apply(
                lambda x: '0+0' if x == 'negative' else x
            )
        
        self.image_dir = os.path.join(data_dir, image_dir)
        self.mask_dir = os.path.join(data_dir, mask_dir)
        self.transform = transform
        self.is_test = is_test
        self.patch_size = patch_size
        self.overlap = overlap
        self.min_tumor_area = min_tumor_area
        
        # Create patch annotations
        self.patch_annotations = self._create_patch_annotations()
        
    def _create_patch_annotations(self) -> List[Dict]:
        """Create patch-level annotations in COCO format."""
        annotations = []
        
        for idx, row in self.df.iterrows():
            img_id = row['image_id']
            isup_grade = row['isup_grade']
            
            # Extract patches from WSI
            patches = self._extract_patches_from_wsi(img_id, isup_grade)
            annotations.extend(patches)
            
        print(f"Created {len(annotations)} patch annotations")
        return annotations
    
    def _extract_patches_from_wsi(self, img_id: str, isup_grade: int) -> List[Dict]:
        """Extract patches from a whole slide image with annotations."""
        patches = []
        
        img_path = os.path.join(self.image_dir, f"{img_id}.tiff")
        mask_path = os.path.join(self.mask_dir, f"{img_id}_mask.tiff")
        
        if not os.path.exists(img_path):
            print(f"[WARNING] Image not found: {img_path}")
            return patches
            
        try:
            # Open WSI
            slide = openslide.OpenSlide(img_path)
            w, h = slide.dimensions
            
            # Open mask if available
            mask_slide = None
            if os.path.exists(mask_path) and not self.is_test:
                try:
                    mask_slide = openslide.OpenSlide(mask_path)
                except:
                    print(f"[WARNING] Could not open mask: {mask_path}")
            
            # Calculate patch coordinates
            step_size = self.patch_size - self.overlap
            
            for y in range(0, h - self.patch_size + 1, step_size):
                for x in range(0, w - self.patch_size + 1, step_size):
                    
                    # Extract patch from image
                    patch = slide.read_region(
                        (x, y), 0, (self.patch_size, self.patch_size)
                    ).convert('RGB')
                    patch_array = np.array(patch)
                    
                    # Skip mostly white/empty patches
                    if self._is_background_patch(patch_array):
                        continue
                    
                    # Create patch annotation
                    patch_annotation = {
                        'image_id': f"{img_id}_{x}_{y}",
                        'original_image_id': img_id,
                        'patch_coords': (x, y),
                        'image': patch_array,
                        'isup_grade': isup_grade,
                        'annotations': []
                    }
                    
                    # Extract mask annotations if available
                    if mask_slide is not None:
                        mask_patch = mask_slide.read_region(
                            (x, y), 0, (self.patch_size, self.patch_size)
                        ).convert('RGB')
                        mask_array = np.array(mask_patch)
                        
                        # Convert mask to binary (red channel > 0)
                        binary_mask = (mask_array[:, :, 0] > 0).astype(np.uint8)
                        
                        # Find connected components (tumor regions)
                        annotations = self._mask_to_annotations(
                            binary_mask, isup_grade
                        )
                        patch_annotation['annotations'] = annotations
                    
                    patches.append(patch_annotation)
            
            slide.close()
            if mask_slide:
                mask_slide.close()
                
        except Exception as e:
            print(f"[ERROR] Failed to process {img_path}: {e}")
            
        return patches
    
    def _is_background_patch(self, patch: np.ndarray, white_threshold: float = 0.8) -> bool:
        """Check if patch is mostly background (white/empty)."""
        # Convert to grayscale
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # Calculate percentage of white pixels
        white_pixels = np.sum(gray > 240)
        total_pixels = gray.shape[0] * gray.shape[1]
        white_ratio = white_pixels / total_pixels
        
        return white_ratio > white_threshold
    
    def _mask_to_annotations(self, mask: np.ndarray, isup_grade: int) -> List[Dict]:
        """Convert binary mask to COCO-style annotations."""
        annotations = []
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(mask)
        
        for label_id in range(1, num_labels):  # Skip background (0)
            # Create binary mask for this component
            component_mask = (labels == label_id).astype(np.uint8)
            
            # Skip small components
            area = np.sum(component_mask)
            if area < self.min_tumor_area:
                continue
            
            # Find contours
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                continue
                
            # Get largest contour
            contour = max(contours, key=cv2.contourArea)
            
            # Convert contour to polygon
            contour = contour.flatten()
            if len(contour) < 6:  # Need at least 3 points
                continue
            
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour.reshape(-1, 1, 2))
            
            # Create annotation
            annotation = {
                'bbox': [x, y, w, h],
                'bbox_mode': BoxMode.XYWH_ABS,
                'segmentation': [contour.tolist()],
                'category_id': isup_grade,  # Use ISUP grade as category
                'area': area,
                'iscrowd': 0
            }
            
            annotations.append(annotation)
        
        return annotations
    
    def __len__(self) -> int:
        return len(self.patch_annotations)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a patch with annotations in Detectron2 format."""
        patch_data = self.patch_annotations[idx]
        
        # Get image
        image = patch_data['image'].copy()
        annotations = patch_data['annotations'].copy()
        
        # Apply transforms
        if self.transform:
            # Prepare for albumentations
            bboxes = []
            category_ids = []
            masks = []
            
            for ann in annotations:
                bbox = ann['bbox']
                # Convert XYWH to albumentations format (normalized)
                x, y, w, h = bbox
                bboxes.append([x, y, x + w, y + h])  # Convert to XYXY
                category_ids.append(ann['category_id'])
                
                # Create mask from segmentation
                if 'segmentation' in ann:
                    mask = self._segmentation_to_mask(
                        ann['segmentation'], image.shape[:2]
                    )
                    masks.append(mask)
            
            # Apply transforms
            if bboxes:
                transformed = self.transform(
                    image=image,
                    bboxes=bboxes,
                    category_ids=category_ids,
                    masks=masks
                )
                
                image = transformed['image']
                bboxes = transformed['bboxes']
                category_ids = transformed['category_ids']
                masks = transformed.get('masks', masks)
                
                # Update annotations
                for i, ann in enumerate(annotations):
                    if i < len(bboxes):
                        x1, y1, x2, y2 = bboxes[i]
                        ann['bbox'] = [x1, y1, x2 - x1, y2 - y1]  # Back to XYWH
                        ann['category_id'] = category_ids[i]
                        
                        if i < len(masks):
                            # Update segmentation from mask
                            ann['segmentation'] = self._mask_to_segmentation(masks[i])
            else:
                # No annotations, just transform image
                transformed = self.transform(image=image)
                image = transformed['image']
        
        # Convert to Detectron2 format
        record = {
            'image_id': patch_data['image_id'],
            'file_name': patch_data['image_id'],  # Virtual file name
            'height': image.shape[0],
            'width': image.shape[1],
            'image': torch.as_tensor(image.transpose(2, 0, 1).astype("float32")),
            'annotations': annotations
        }
        
        return record
    
    def _segmentation_to_mask(self, segmentation: List, image_shape: Tuple[int, int]) -> np.ndarray:
        """Convert segmentation polygon to binary mask."""
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        for seg in segmentation:
            poly = np.array(seg).reshape(-1, 2)
            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
            
        return mask
    
    def _mask_to_segmentation(self, mask: np.ndarray) -> List[List[float]]:
        """Convert binary mask to segmentation polygon."""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        segmentations = []
        for contour in contours:
            if len(contour) >= 3:
                segmentation = contour.flatten().tolist()
                segmentations.append(segmentation)
                
        return segmentations


def get_detection_transforms(is_training: bool = True, img_size: int = 1024) -> A.Compose:
    """Get augmentation transforms for detection."""
    
    if is_training:
        return A.Compose([
            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            
            # Color transforms (important for histology)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0
                ),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
            ], p=0.5),
            
            # Advanced transforms
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1.0),
            ], p=0.3),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            
            # Ensure size
            A.Resize(img_size, img_size, always_apply=True),
            
            # Normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['category_ids'],
            min_visibility=0.3
        ))
    else:
        return A.Compose([
            A.Resize(img_size, img_size, always_apply=True),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['category_ids'],
            min_visibility=0.3
        ))
