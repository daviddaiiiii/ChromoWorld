import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
from typing import Optional, List, Dict, Tuple, Union
import glob
import random
from scipy import interpolate  # Add this import for smooth interpolation
import torchvision.transforms.functional as TF
from scipy.ndimage import distance_transform_edt, gaussian_filter  # For PSF blur
from skimage import morphology  # For skeleton extraction in PSF augmentation
from PIL import ImageOps

class OpenKaryo2DDataset(Dataset):
    """
    OpenKaryo Dataset Dataloader for YOLO format with class-based sampling.
    
    Args:
        label_path (str): Path to label folder containing txt files
        image_path (str): Path to image folder
        visualize_path (str): Path to visualization folder
        transform (callable, optional): Transform to apply to images
        resample (int): Number of points to resample. Default 32.
            - If resample <= 0: mask returns original polygon
            - If resample > 0: mask returns N resampled points
        data_augmentation (bool): Whether to apply data augmentation (default: True)
    """
    
    def __init__(self, 
                 label_path: str,
                 image_path: str,
                 visualize_path: str,
                 transform: Optional[callable] = None,
                 resample: int = 32,
                 data_augmentation: bool = True,
                 invert_colormap: bool = True,
                 return_skeleton_image: bool = False,
                 ):
        '''
        Args:
            label_path (str): Path to label folder containing txt files
            image_path (str): Path to image folder
            visualize_path (str): Path to visualization folder
            transform (callable, optional): Transform to apply to images
            resample (int): Number of points to resample. Default 32.
                - If resample <= 0: mask returns original polygon
                - If resample > 0: mask returns N resampled points
            data_augmentation (bool): Whether to apply data augmentation (default: True)
            return_skeleton_image (bool): Whether to compute and return skeleton_image (default: False, slow operation)
        '''
        
        self.label_path = label_path
        self.image_path = image_path
        self.visualize_path = visualize_path
        self.transform = transform
        self.resample = resample
        self.data_augmentation = data_augmentation
        self.invert_colormap = invert_colormap
        self.return_skeleton_image = return_skeleton_image
        # Store original image dimensions for scaling
        self.original_width = None
        self.original_height = None
        
        # Load all samples and create class-to-sample mapping
        self.samples = self._load_all_samples()
        self.class_to_samples = self._create_class_mapping()
        
        # Create flat list of all class instances for proper indexing
        self.all_instances = self._create_instance_list()
        
        print(f"Total samples (sum of all class instances): {len(self.all_instances)}")
        print(f"Available classes: {sorted(list(self.class_to_samples.keys()))}")
    
    def _load_all_samples(self) -> List[Dict]:
        """Load all samples from YOLO format txt files."""
        samples = []
        
        # Search for all txt files in the label path recursively
        txt_files = []
        for root, dirs, files in os.walk(self.label_path):
            for file in files:
                if file.endswith('.txt'):
                    txt_files.append(os.path.join(root, file))
        
        for txt_file in txt_files:
            # Extract base filename (e.g., "Banding-0000" from "Banding-0000.txt")
            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            
            # Construct image and visualize paths
            image_file = os.path.join(self.image_path, f"{base_name}.png")
            visualize_file = os.path.join(self.visualize_path, f"{base_name}.png")
            
            # Check if corresponding files exist
            if os.path.exists(image_file):
                # Parse annotations to get class IDs
                annotations = self._parse_yolo_label(txt_file)
                class_ids = set()
                bbox_annotations = []
                mask_annotations = []
                
                for ann in annotations:
                    class_id = ann['class_id']
                    class_ids.add(class_id)
                    if ann['type'] == 'bbox':
                        bbox_annotations.append(ann)
                    elif ann['type'] == 'mask':
                        mask_annotations.append(ann)
            
                # Only include samples that have valid annotations (bbox or mask)
                if class_ids and (bbox_annotations or mask_annotations):
                    sample = {
                        'label_file': txt_file,
                        'image_file': image_file,
                        'visualize_file': visualize_file if os.path.exists(visualize_file) else None,
                        'base_name': base_name,
                        'class_ids': list(class_ids),
                        'bbox_annotations': bbox_annotations,
                        'mask_annotations': mask_annotations
                    }
                    samples.append(sample)
        
        return samples
    
    def _create_class_mapping(self) -> Dict[int, List[Dict]]:
        """Create mapping from class ID to list of samples containing that class."""
        class_to_samples = {}
        
        for sample in self.samples:
            for class_id in sample['class_ids']:
                if class_id not in class_to_samples:
                    class_to_samples[class_id] = []
                class_to_samples[class_id].append(sample)
        
        return class_to_samples
    
    def _create_instance_list(self) -> List[Dict]:
        """Create a flat list of all class instances for proper indexing.
        
        Note: All annotations are stored here (not filtered by class_id).
        Filtering by class_id is done AFTER data augmentation in __getitem__
        to allow augmentations that need all instance masks (e.g., copy-paste).
        """
        instances = []
        
        for sample in self.samples:
            for class_id in sample['class_ids']:
                # Check if this class has at least one annotation
                bbox_for_class = [ann for ann in sample['bbox_annotations'] if ann['class_id'] == class_id]
                mask_for_class = [ann for ann in sample['mask_annotations'] if ann['class_id'] == class_id]
                
                # Skip if this class has no annotations at all
                if not bbox_for_class and not mask_for_class:
                    continue
                
                # Store ALL annotations, not just filtered ones
                # Filtering by class_id happens after augmentation
                instance = {
                    'sample': sample,
                    'class_id': class_id,
                    'all_bbox_annotations': sample['bbox_annotations'],  # All annotations
                    'all_mask_annotations': sample['mask_annotations'],  # All annotations
                }
                instances.append(instance)
        
        return instances
    
    def _parse_yolo_label(self, label_file: str) -> List[Dict]:
        """Parse YOLO format label file - handle both bbox and mask formats."""
        annotations = []
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:  # At least class_id + 4 coordinates
                continue
            
            class_id = int(parts[0])
            coordinates = [float(x) for x in parts[1:]]
            
            if len(coordinates) == 4:
                # YOLO bbox format: class_id x_center y_center width height
                x_center, y_center, width, height = coordinates
                annotation = {
                    'class_id': class_id,
                    'bbox': [x_center, y_center, width, height],
                    'type': 'bbox'
                }
                annotations.append(annotation)
            
            elif len(coordinates) >= 6:
                # YOLO mask format: class_id x1 y1 x2 y2 ... (polygon points)
                polygon = coordinates
                annotation = {
                    'class_id': class_id,
                    'polygon': polygon,
                    'type': 'mask'
                }
                annotations.append(annotation)
        
        return annotations
    
    def _create_mask_image(self, annotations: List[Dict], image_size: Tuple[int, int]) -> torch.Tensor:
        """Create binary mask image from mask annotations."""
        height, width = image_size
        mask = torch.zeros((height, width), dtype=torch.uint8)

        from PIL import Image, ImageDraw

        # Create a PIL image to draw all polygons
        mask_pil = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask_pil)

        for ann in annotations:
            if 'polygon' in ann:
                polygon = ann['polygon']
                points = []
                for i in range(0, len(polygon), 2):
                    if i + 1 < len(polygon):
                        x = int(polygon[i] * width)
                        y = int(polygon[i + 1] * height)
                        points.append((x, y))
                if len(points) >= 3:
                    draw.polygon(points, fill=1)

        mask_pil_array = np.array(mask_pil)
        mask = torch.from_numpy(mask_pil_array)

        return mask
    
    
    def _create_bbox_from_mask(self, annotations: List[Dict], image_size: Tuple[int, int]) -> List[Dict]:
        """Create bbox from mask annotations in YOLO format."""
        height, width = image_size
        bbox_list = []
        
        # Create bbox for each polygon, YOLO format
        for ann in annotations: 
            if 'polygon' in ann:
                polygon = ann['polygon']
                points = [] # extract all points
                for i in range(0, len(polygon), 2):
                    if i + 1 < len(polygon):
                        x = int(polygon[i] * width)
                        y = int(polygon[i + 1] * height)
                        points.append((x, y)) # extract all points
                
                if len(points) >= 3:
                    # Find min/max coordinates to create proper bounding box
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    
                    x_min = min(x_coords)
                    x_max = max(x_coords)
                    y_min = min(y_coords)
                    y_max = max(y_coords)
                    
                    # Convert to YOLO format (normalized center coordinates and dimensions)
                    x_center = (x_min + x_max) / 2.0 / width
                    y_center = (y_min + y_max) / 2.0 / height
                    bbox_width = (x_max - x_min) / width
                    bbox_height = (y_max - y_min) / height
                    
                    bbox_list.append([x_center, y_center, bbox_width, bbox_height])
            
        return bbox_list
    
    def _scale_annotations(self, annotations: List[Dict], width_height_ratio: float) -> List[Dict]:
        """Scale annotations according to image transform."""
        scaled_annotations = []
        
        for ann in annotations:
            scaled_ann = ann.copy()
            
            if ann['type'] == 'bbox':
                # Scale bbox coordinates
                x_center, y_center, width, height = ann['bbox']
                scaled_ann['bbox'] = [
                    x_center * width_height_ratio,
                    y_center,
                    width * width_height_ratio,
                    height
                ]
            
            elif ann['type'] == 'mask':
                # Scale polygon coordinates
                polygon = ann['polygon']
                scaled_polygon = []
                for i in range(0, len(polygon), 2):
                    if i + 1 < len(polygon):
                        x, y = polygon[i], polygon[i + 1]
                        scaled_polygon.extend([x * width_height_ratio, y])
                scaled_ann['polygon'] = scaled_polygon
            
            scaled_annotations.append(scaled_ann)
        
        return scaled_annotations
    
    def _resample_contour_smooth(self, polygon: List[float], num_points: int) -> List[float]:
        """
        Resample polygon contour to fixed number of points using smooth spline interpolation.
        
        Args:
            polygon: List of coordinates [x1, y1, x2, y2, ...]
            num_points: Number of points to resample to
            
        Returns:
            resampled_polygon: List of resampled coordinates with fixed length
        """
        if len(polygon) < 6:  # Need at least 3 points
            # Pad with repeated points if too short
            while len(polygon) < num_points * 2:
                polygon.extend(polygon[:2])
            return polygon[:num_points * 2]
        
        # Convert to array of points (N, 2)
        points = np.array(polygon).reshape(-1, 2)
        
        # Remove duplicate consecutive points
        if len(points) > 1:
            # Keep only points that are different from the previous one
            mask = np.ones(len(points), dtype=bool)
            mask[1:] = np.any(np.abs(np.diff(points, axis=0)) > 1e-6, axis=1)
            points = points[mask]
        
        # Check if we have enough unique points
        if len(points) < 2:
            # If not enough points, create a simple polygon
            center = points[0] if len(points) > 0 else np.array([0.5, 0.5])
            radius = 0.01
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            points = np.array([center + radius * np.array([np.cos(a), np.sin(a)]) for a in angles])
        
        # Close the contour if not already closed
        if not np.allclose(points[0], points[-1], atol=1e-6):
            points = np.vstack([points, points[0:1]])
        
        # Calculate cumulative arc length for parameterization
        segment_lengths = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_lengths[-1]
        
        # Handle degenerate case: if total_length is zero or too small, use uniform sampling
        if total_length < 1e-6:
            # Degenerate polygon (all points are the same or very close)
            # Use uniform parameterization
            t = np.linspace(0, 1, len(points), endpoint=False)
        else:
            # Normalize to [0, 1] for better numerical stability
            t = cumulative_lengths / total_length
        
        # Use spline interpolation for smooth resampling
        # Use periodic boundary conditions since contour is closed
        try:
            # Check if we have valid parameterization
            if np.any(np.isnan(t)) or np.any(np.isinf(t)) or len(points) < 3:
                raise ValueError("Invalid parameterization")
            
            # Cubic spline interpolation with periodic boundary
            tck_x, _ = interpolate.splprep([points[:, 0]], u=t, s=0, per=1, k=min(3, len(points)-1))
            tck_y, _ = interpolate.splprep([points[:, 1]], u=t, s=0, per=1, k=min(3, len(points)-1))
            
            # Generate uniform sample points
            t_new = np.linspace(0, 1, num_points, endpoint=False)
            
            # Evaluate spline at new points
            resampled_x = interpolate.splev(t_new, tck_x[0])
            resampled_y = interpolate.splev(t_new, tck_y[0])
        except (ValueError, TypeError, np.linalg.LinAlgError) as e:
            # Fallback to linear interpolation if spline fails
            # If t is invalid, use uniform sampling
            if np.any(np.isnan(t)) or np.any(np.isinf(t)) or total_length < 1e-6:
                # Uniform sampling along the polygon
                indices = np.linspace(0, len(points) - 1, num_points, endpoint=False).astype(int)
                resampled_x = points[indices, 0]
                resampled_y = points[indices, 1]
            else:
                # Linear interpolation
                t_new = np.linspace(0, 1, num_points, endpoint=False)
                resampled_x = np.interp(t_new, t, points[:, 0])
                resampled_y = np.interp(t_new, t, points[:, 1])
        
        # Flatten back to [x1, y1, x2, y2, ...] format
        resampled_polygon = np.stack([resampled_x, resampled_y], axis=1).flatten().tolist()
        
        return resampled_polygon
    
    def _resample_mask_annotations(self, mask_annotations: List[Dict]) -> List[Dict]:
        """
        Resample all polygons in mask annotations to fixed number of points.
        Only called when self.resample > 0.
        
        Args:
            mask_annotations: List of mask annotation dicts
            
        Returns:
            resampled_annotations: List of resampled mask annotation dicts
        """
        
        resampled_annotations = []
        for ann in mask_annotations:
            if 'polygon' in ann:
                # Resample the polygon
                resampled_polygon = self._resample_contour_smooth(
                    ann['polygon'], 
                    self.resample
                )
                # Create new annotation with resampled polygon
                resampled_ann = ann.copy()
                resampled_ann['polygon'] = resampled_polygon
                resampled_annotations.append(resampled_ann)
            else:
                resampled_annotations.append(ann)
        
        return resampled_annotations
    
    def _apply_invert_colormap(self, image: Image.Image) -> Image.Image:
        """
        Apply colormap inversion to image (doesn't affect annotations).
        Inverts the image colors: dark becomes light, light becomes dark.
        Useful for simulating different staining/imaging conditions.
        """
        return ImageOps.invert(image)
    
    def _apply_horizontal_flip(self, image: Image.Image, bbox_annotations: List[Dict], 
                               mask_annotations: List[Dict]) -> Tuple[Image.Image, List[Dict], List[Dict]]:
        """Apply horizontal flip to image and annotations."""
        image = TF.hflip(image)
        
        # Flip bbox annotations
        flipped_bbox = []
        for ann in bbox_annotations:
            x_center, y_center, width, height = ann['bbox']
            # Flip x coordinate: x_new = 1 - x_old
            x_center_new = 1.0 - x_center
            flipped_ann = ann.copy()
            flipped_ann['bbox'] = [x_center_new, y_center, width, height]
            flipped_bbox.append(flipped_ann)
        
        # Flip mask annotations
        flipped_mask = []
        for ann in mask_annotations:
            polygon = ann['polygon']
            flipped_polygon = []
            for i in range(0, len(polygon), 2):
                if i + 1 < len(polygon):
                    x, y = polygon[i], polygon[i + 1]
                    # Flip x coordinate
                    x_new = 1.0 - x
                    flipped_polygon.extend([x_new, y])
            flipped_ann = ann.copy()
            flipped_ann['polygon'] = flipped_polygon
            flipped_mask.append(flipped_ann)
        
        return image, flipped_bbox, flipped_mask
    
    def _apply_vertical_flip(self, image: Image.Image, bbox_annotations: List[Dict], 
                            mask_annotations: List[Dict]) -> Tuple[Image.Image, List[Dict], List[Dict]]:
        """Apply vertical flip to image and annotations."""
        image = TF.vflip(image)
        
        # Flip bbox annotations
        flipped_bbox = []
        for ann in bbox_annotations:
            x_center, y_center, width, height = ann['bbox']
            # Flip y coordinate: y_new = 1 - y_old
            y_center_new = 1.0 - y_center
            flipped_ann = ann.copy()
            flipped_ann['bbox'] = [x_center, y_center_new, width, height]
            flipped_bbox.append(flipped_ann)
        
        # Flip mask annotations
        flipped_mask = []
        for ann in mask_annotations:
            polygon = ann['polygon']
            flipped_polygon = []
            for i in range(0, len(polygon), 2):
                if i + 1 < len(polygon):
                    x, y = polygon[i], polygon[i + 1]
                    # Flip y coordinate
                    y_new = 1.0 - y
                    flipped_polygon.extend([x, y_new])
            flipped_ann = ann.copy()
            flipped_ann['polygon'] = flipped_polygon
            flipped_mask.append(flipped_ann)
        
        return image, flipped_bbox, flipped_mask
    
    def _apply_rotation(self, image: Image.Image, bbox_annotations: List[Dict], 
                       mask_annotations: List[Dict], angle: float) -> Tuple[Image.Image, List[Dict], List[Dict]]:
        """Apply rotation to image and annotations."""
        # Rotate image
        image = TF.rotate(image, angle, expand=False, fill=0)
        
        # Convert angle to radians
        # Note: TF.rotate uses counter-clockwise rotation (positive angle)
        # In image coordinates (y-axis down), we need to adjust the rotation matrix
        angle_rad = np.radians(-angle)  # Negate angle to match image coordinate system
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotation matrix around center (0.5, 0.5) in image coordinates
        def rotate_point(x, y):
            # Translate to origin (center of image)
            x_t = x - 0.5
            y_t = y - 0.5
            # Apply rotation (adjusted for image coordinate system)
            x_rot = x_t * cos_a - y_t * sin_a
            y_rot = x_t * sin_a + y_t * cos_a
            # Translate back
            x_new = x_rot + 0.5
            y_new = y_rot + 0.5
            return x_new, y_new
        
        # Rotate bbox annotations (convert to polygon, rotate, then get bbox)
        rotated_bbox = []
        for ann in bbox_annotations:
            x_center, y_center, width, height = ann['bbox']
            # Get 4 corners of bbox
            x_min = x_center - width / 2
            x_max = x_center + width / 2
            y_min = y_center - height / 2
            y_max = y_center + height / 2
            
            corners = [
                (x_min, y_min), (x_max, y_min),
                (x_max, y_max), (x_min, y_max)
            ]
            
            # Rotate corners
            rotated_corners = [rotate_point(x, y) for x, y in corners]
            
            # Get new bbox from rotated corners
            xs = [p[0] for p in rotated_corners]
            ys = [p[1] for p in rotated_corners]
            x_min_new = max(0.0, min(xs))
            x_max_new = min(1.0, max(xs))
            y_min_new = max(0.0, min(ys))
            y_max_new = min(1.0, max(ys))
            
            x_center_new = (x_min_new + x_max_new) / 2
            y_center_new = (y_min_new + y_max_new) / 2
            width_new = x_max_new - x_min_new
            height_new = y_max_new - y_min_new
            
            rotated_ann = ann.copy()
            rotated_ann['bbox'] = [x_center_new, y_center_new, width_new, height_new]
            rotated_bbox.append(rotated_ann)
        
        # Rotate mask annotations
        rotated_mask = []
        for ann in mask_annotations:
            polygon = ann['polygon']
            rotated_polygon = []
            for i in range(0, len(polygon), 2):
                if i + 1 < len(polygon):
                    x, y = polygon[i], polygon[i + 1]
                    x_new, y_new = rotate_point(x, y)
                    # Clip to valid range
                    x_new = max(0.0, min(1.0, x_new))
                    y_new = max(0.0, min(1.0, y_new))
                    rotated_polygon.extend([x_new, y_new])
            rotated_ann = ann.copy()
            rotated_ann['polygon'] = rotated_polygon
            rotated_mask.append(rotated_ann)
        
        return image, rotated_bbox, rotated_mask
    
    def _apply_color_jitter(self, image: Image.Image) -> Image.Image:
        """Apply color jitter to image only (doesn't affect annotations)."""
        # Random brightness
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)
        
        # Random contrast
        if random.random() > 0.5:
            contrast_factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_factor)
        
        # Random saturation
        if random.random() > 0.5:
            saturation_factor = random.uniform(0.8, 1.2)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation_factor)
        
        return image
    
    def _apply_copy_paste(self, image: Image.Image, bbox_annotations: List[Dict], 
                         mask_annotations: List[Dict], all_instances: List[Dict]) -> Tuple[Image.Image, List[Dict], List[Dict]]:
        """
        Apply copy-paste augmentation: copy 0-50% instances from current image 
        and paste them to random positions to increase detection difficulty.
        """
        
        try:
            from PIL import ImageDraw
            
            current_width, current_height = image.size
            
            # Randomly select instances to copy
            instances_to_copy = random.sample(mask_annotations,
                                              random.randint(0, len(mask_annotations)))
            
            new_mask_annotations = []
            
            for mask_ann in instances_to_copy:
                # Create mask for this instance
                instance_mask = Image.new('L', (current_width, current_height), 0)
                draw_mask = ImageDraw.Draw(instance_mask)
                
                polygon = mask_ann['polygon']
                points = []
                for i in range(0, len(polygon), 2):
                    if i + 1 < len(polygon):
                        x = int(polygon[i] * current_width)
                        y = int(polygon[i + 1] * current_height)
                        points.append((x, y))
                
                if len(points) < 3:
                    continue
                
                draw_mask.polygon(points, fill=255)
                
                # Get bounding box of the instance
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Crop the instance region from both image and mask
                bbox_crop = (x_min, y_min, x_max, y_max)
                instance_crop = image.crop(bbox_crop)
                mask_crop = instance_mask.crop(bbox_crop)
                
                crop_width = x_max - x_min
                crop_height = y_max - y_min
                
                # Generate random paste position
                # Ensure the pasted region is within image bounds
                max_x = current_width - crop_width
                max_y = current_height - crop_height
                
                if max_x > 0 and max_y > 0:
                    paste_x = random.randint(0, max_x)
                    paste_y = random.randint(0, max_y)
                    
                    # Paste the instance to new position
                    image.paste(instance_crop, (paste_x, paste_y), mask_crop)
                    
                    # Calculate offset for polygon coordinates
                    offset_x = (paste_x - x_min) / current_width
                    offset_y = (paste_y - y_min) / current_height
                    
                    # Create new annotation with translated polygon
                    new_polygon = []
                    for i in range(0, len(polygon), 2):
                        if i + 1 < len(polygon):
                            x_norm = polygon[i]
                            y_norm = polygon[i + 1]
                            # Translate coordinates
                            x_new = x_norm + offset_x
                            y_new = y_norm + offset_y
                            # Clip to valid range
                            x_new = max(0.0, min(1.0, x_new))
                            y_new = max(0.0, min(1.0, y_new))
                            new_polygon.extend([x_new, y_new])
                    
                    # Add new annotation
                    new_ann = mask_ann.copy()
                    new_ann['polygon'] = new_polygon
                    new_mask_annotations.append(new_ann)
            
            # Add new pasted annotations to the list
            mask_annotations.extend(new_mask_annotations)
            
        except Exception as e:
            # If copy-paste fails, just return original
            print(f"Copy-paste augmentation failed: {e}")
            pass
        
        return image, bbox_annotations, mask_annotations

    def _create_shape_image(self, mask_annotations: List[Dict], width: int, height: int) -> Image.Image:
        """
        Create pseudo-binarized image by drawing mask_annotations (after augment, before PSF).
        Used for shape-based SSL: no texture, only shape from masks.
        """
        from PIL import ImageDraw
        shape_pil = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(shape_pil)
        for ann in mask_annotations:
            if 'polygon' in ann and len(ann['polygon']) >= 6:
                polygon = ann['polygon']
                points = [(int(polygon[i] * width), int(polygon[i + 1] * height)) for i in range(0, len(polygon) - 1, 2)]
                if len(points) >= 3:
                    draw.polygon(points, fill=255)
        return shape_pil.convert('RGB')

    def _apply_psf_blur(self, image: Image.Image, mask_annotations: List[Dict]) -> Image.Image:
            """
            [Optimized] Apply Point Spread Function (PSF) blur augmentation.
            
            Optimization:
            1. Draws all masks onto a single canvas at once (removes loop overhead).
            2. Uses Distance Transform instead of skeletonize (10x-50x faster).
            3. Vectorized Gaussian blur.
            """
            # Convert image to numpy array
            img_array = np.array(image).astype(np.float32)
            height, width = img_array.shape[:2]
            is_rgb = len(img_array.shape) > 2
            
            # 1. Base Blur: Simulate optical system imperfection (background blur)
            # Use a smaller sigma for the base image
            base_sigma = random.uniform((height/384)*(width/384), (height/192)*(width/192)) 
            base_blurred = gaussian_filter(img_array, sigma=base_sigma if not is_rgb else (base_sigma, base_sigma, 0))
            
            if not mask_annotations:
                return Image.fromarray(np.clip(base_blurred, 0, 255).astype(np.uint8))

            # 2. Fast Mask Generation: Draw all polygons onto ONE mask
            # Using PIL.ImageDraw is fast enough if we don't create new images in a loop
            from PIL import ImageDraw
            full_mask_pil = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(full_mask_pil)
            
            has_valid_polygon = False
            for ann in mask_annotations:
                if 'polygon' in ann and len(ann['polygon']) >= 6: # at least 3 points
                    polygon = ann['polygon']
                    # Convert normalized coords to pixels
                    points = []
                    for i in range(0, len(polygon), 2):
                        points.append((int(polygon[i] * width), int(polygon[i+1] * height)))
                    draw.polygon(points, fill=1)
                    has_valid_polygon = True
            
            if not has_valid_polygon:
                return Image.fromarray(np.clip(base_blurred, 0, 255).astype(np.uint8))

            # Convert mask to numpy
            # mask_array has 1s inside chromosomes, 0s outside
            mask_array = np.array(full_mask_pil)
            
            # 3. Fast Skeleton Approximation using Distance Transform
            # Instead of iterative skeletonize, we compute distance from background.
            # The "center" of the chromosome has the highest distance.
            dist_map = distance_transform_edt(mask_array)
            
            # Heuristic: We only want the "core" to glow. 
            # We can threshold the distance map to keep only the inner-most pixels.
            # E.g., keep pixels that are > 1 pixel away from the border.
            # For very thin objects, this might be aggressive, so we use max(1, ...)
            # A simple 'dist_map > 1' acts like a fast erosion.
            skeleton_approx = (dist_map > 1.5).astype(np.float32)
            
            # Fallback for very thin structures: if erosion killed everything, use the mask itself
            if skeleton_approx.sum() < 10:
                skeleton_approx = mask_array.astype(np.float32)

            # 4. Create the "Glow" (PSF Kernel)
            # Blur the skeleton heavily to simulate dCas9 glow
            instance_sigma = random.uniform((height/128)*(width/128), (height/64)*(width/64))
            
            # Blur the skeleton to create the light distribution map
            psf_glow_map = gaussian_filter(skeleton_approx, sigma=instance_sigma)
            
            # Normalize glow map to [0, 1] range to use as a blend weight
            # We perform a robust normalization to avoid outliers
            glow_max = psf_glow_map.max()
            if glow_max > 0:
                psf_glow_map = psf_glow_map / glow_max
                # Enhance the glow curve (gamma correction style) to make it "pop"
                psf_glow_map = np.power(psf_glow_map, 0.7) 
            
            # 5. Blend
            # Logic: The skeleton area is dominated by "Instance Blur" (the glow), 
            # the background is dominated by "Base Blur".
            
            # We also want the "Instance Blur" version of the image to be brighter usually
            # because dCas9 adds light.
            instance_layer = gaussian_filter(img_array, sigma=instance_sigma if not is_rgb else (instance_sigma, instance_sigma, 0))
            
            # Make the instance layer brighter? Optional.
            # instance_layer = instance_layer * 1.2 
            
            if is_rgb:
                weight = psf_glow_map[:, :, np.newaxis]
            else:
                weight = psf_glow_map
                
            # Linear Interpolation blending
            final_image = base_blurred * (1 - weight) + instance_layer * weight
            
            return Image.fromarray(np.clip(final_image, 0, 255).astype(np.uint8))

    def _create_skeleton_image(self, mask_annotations: List[Dict], image_size: Tuple[int, int], 
                                thickness: int = 5) -> torch.Tensor:
        """
        [Optimized] Create skeleton image using Distance Transform.
        Much faster than morphological skeletonization.
        """
        height, width = image_size
        
        # 1. Draw all masks at once
        from PIL import ImageDraw
        mask_pil = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask_pil)
        
        for ann in mask_annotations:
            if 'polygon' in ann and len(ann['polygon']) >= 6:
                polygon = ann['polygon']
                points = []
                for i in range(0, len(polygon), 2):
                    points.append((int(polygon[i] * width), int(polygon[i+1] * height)))
                draw.polygon(points, fill=1)
                
        mask_array = np.array(mask_pil)
        if mask_array.max() == 0:
            return torch.zeros((height, width), dtype=torch.uint8)

        # 2. Use Distance Transform to find the centerline
        dist_map = distance_transform_edt(mask_array)
        
        # 3. Create a clean centerline
        # Logic: Pixels that are strictly greater than their neighbors in at least one direction
        # But a simpler heuristic for visualization is just high distance values.
        # However, to get a nice line, we can use "Ridge Detection" or simply:
        # Since we want a skeleton of specific thickness, we can threshold the distance map
        # differently or just use the mask structure.
        
        # For visualization purposes (and since we want thickness), 
        # let's effectively "erode" the shape to its center based on thickness.
        # This is a hack: if an object is 10px wide, max dist is 5. 
        # If we want a 3px wide skeleton, we keep pixels where dist > (max_dist - 1.5).
        # But objects have different widths.
        
        # FASTEST Robust Method for "Skeleton-like" visualization without strict skeletonize:
        # Heavy erosion.
        from scipy.ndimage import binary_erosion
        
        # Erode until it's thin. This simulates skeletonization iteration but efficiently.
        # But since we want specific thickness, we can just return the centers.
        
        # Let's stick to distance transform for speed.
        # We can find local maxima in the distance transform, but that's complex.
        
        # Reverting to a faster approximation:
        # 1. Distance Transform
        # 2. Keep pixels where dist_map > 0.
        # 3. If we want "thin", we want the ridges.
        # If speed is paramount and exact topology doesn't matter for viz:
        
        # Let's optimize the original logic (Dilated Skeleton) using distance transform
        # Ideally, Distance Transform Ridge is best.
        # Approximation: Top % of distance values per connected component? Too slow.
        
        # Alternative: Just use the distance map itself as a "Soft Skeleton"!
        # It looks nicer for visualization anyway.
        # But if you strictly need a binary mask of thickness T:
        
        # Let's use the scikit-image medial_axis if available (faster than skeletonize),
        # Or just stick to the original skeletonize but ONLY DO IT ONCE on the full mask.
        # Doing it once on 512x512 is acceptable (approx 0.05s). 
        # Doing it 46 times was the problem.
        
        try:
            # Batch skeletonize: Process the whole mask at once
            skeleton = morphology.skeletonize(mask_array)
            skeleton = skeleton.astype(np.float32)
            
            # Dilate if needed
            if thickness > 1:
                # Use distance transform for dilation (faster than morphology.dilation with disk)
                # Dilation of a set X is roughly {p | dist(p, X) <= radius}
                # So we calculate distance from the *skeleton*
                
                # Invert skeleton to compute distance from it
                # distance_transform_edt computes distance to nearest zero.
                # So we put 0 where skeleton is, 1 elsewhere.
                skel_dist = distance_transform_edt(1 - skeleton)
                
                # Keep pixels close to skeleton
                final_skel = (skel_dist <= (thickness / 2.0)).astype(np.float32)
            else:
                final_skel = skeleton

        except Exception:
            final_skel = np.zeros_like(mask_array, dtype=np.float32)
            
        return torch.from_numpy((final_skel * 255).astype(np.uint8))
    
    def __len__(self) -> int:
        return len(self.all_instances)
    
    def __getitem__(self, idx: int) -> Dict:
        # Get the instance at this index
        instance = self.all_instances[idx]
        sample = instance['sample']
        class_id = instance['class_id'] 
        # Use ALL annotations for augmentation (filtering by class_id happens after augmentation)
        bbox_annotations = [ann.copy() for ann in instance['all_bbox_annotations']]
        mask_annotations = [ann.copy() for ann in instance['all_mask_annotations']]
        

        
        # If class_id is -1 (random selection needed), pick from available classes
        if class_id == -1:
            # Build list of valid class IDs (exclude -1)
            class_list = [ann['class_id'] for ann in mask_annotations if ann['class_id'] != -1]
            if class_list:
                class_id = random.choice(class_list)
            else:
                # No valid class IDs available, skip this sample and get next one
                return self.__getitem__((idx + 1) % len(self))
        
        # Load image
        try:
            image = Image.open(sample['image_file']).convert('RGB')
            self.original_width, self.original_height = image.size
        except Exception as e:
            print(f"Error loading image {sample['image_file']}: {e}")
            # Return a dummy sample or skip
            return self.__getitem__((idx + 1) % len(self))
        
        # Load visualize image if available
        visualize_image = None
        if sample.get('visualize_file') and os.path.exists(sample['visualize_file']):
            try:
                visualize_image = Image.open(sample['visualize_file']).convert('RGB')
            except Exception as e:
                print(f"Error loading visualize image {sample['visualize_file']}: {e}")
        
        if self.invert_colormap:
            image = self._apply_invert_colormap(image)
        
        # Apply data augmentation if enabled (before transform)
        # NOTE: Augmentation uses ALL annotations, filtering by class_id happens AFTER
        if self.data_augmentation:
            # Copy-paste augmentation (needs all instance masks)
            image, bbox_annotations, mask_annotations = self._apply_copy_paste(
                image, bbox_annotations, mask_annotations, self.all_instances)
            
            # Random horizontal flip
            if random.random() > 0.5:
                image, bbox_annotations, mask_annotations = self._apply_horizontal_flip(
                    image, bbox_annotations, mask_annotations)
                if visualize_image is not None:
                    visualize_image = TF.hflip(visualize_image)
            
            # Random vertical flip
            if random.random() > 0.5:
                image, bbox_annotations, mask_annotations = self._apply_vertical_flip(
                    image, bbox_annotations, mask_annotations)
                if visualize_image is not None:
                    visualize_image = TF.vflip(visualize_image)
            
            # Random rotation (±30 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-30, 30)
                image, bbox_annotations, mask_annotations = self._apply_rotation(
                    image, bbox_annotations, mask_annotations, angle)
                if visualize_image is not None:
                    visualize_image = TF.rotate(visualize_image, angle, expand=False, fill=0)
        
        # Apply PSF (Point Spread Function) blur augmentation using skeleton on the resized image
        # This uses ALL mask annotations before filtering
        traget_size = self.transform.transforms[0].size
        image = image.resize(traget_size)
        # Shape image: from mask_annotations (after rotate, before PSF), for shape SSL
        shape_pil = self._create_shape_image(mask_annotations, image.size[0], image.size[1])
        psf_image = self._apply_psf_blur(image, mask_annotations)

        # Store all annotations BEFORE filtering only if skeleton_image is needed
        all_mask_annotations_for_skeleton = None
        if self.return_skeleton_image:
            all_mask_annotations_for_skeleton = [ann.copy() for ann in mask_annotations]
        
        # *** FILTER BY CLASS_ID AFTER AUGMENTATION ***
        # Now filter annotations to only keep those matching the target class_id
        bbox_annotations = [ann for ann in bbox_annotations if ann['class_id'] == class_id]
        mask_annotations = [ann for ann in mask_annotations if ann['class_id'] == class_id]
            
        
        # Apply transform to image
        if self.transform:
            # Apply transform
            image = self.transform(image)
            # Apply same transform to psf_image and shape_image
            psf_image = self.transform(psf_image)
            shape_image = self.transform(shape_pil)
        else:
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image = self.transform(image)
            psf_image = self.transform(psf_image)
            shape_image = self.transform(shape_pil)

        if visualize_image:
            visualize_image = self.transform(visualize_image)
        
        # Calculate scale factors
        if hasattr(image, 'shape'):
            # Tensor format
            new_height, new_width = image.shape[-2], image.shape[-1]
        else:
            # PIL format
            new_width, new_height = image.size
        
        width_height_ratio = new_width / new_height
        
        # Scale annotations
        bbox_annotations = self._scale_annotations(bbox_annotations, width_height_ratio)
        mask_annotations = self._scale_annotations(mask_annotations, width_height_ratio)
        # Scale unfiltered annotations for skeleton_image only if needed
        if self.return_skeleton_image and all_mask_annotations_for_skeleton is not None:
            all_mask_annotations_for_skeleton = self._scale_annotations(all_mask_annotations_for_skeleton, width_height_ratio)
        
        # **RESAMPLE MASK ANNOTATIONS** - only if resample > 0
        if self.resample > 0:
            mask_annotations = self._resample_mask_annotations(mask_annotations)
        
        # Prepare return dictionary
        result = {
            'image': image,
            'psf_image': psf_image,  # PSF blur augmented image
            'shape_image': shape_image,  # Pseudo-binarized image for shape SSL
            'base_name': sample['base_name'],
            'class_id': class_id,
        }
        
        if visualize_image is not None:
            result['visualize_image'] = visualize_image
        
        # Return different formats based on annotation type
        # Priority: mask > bbox (mask provides more information)
        if mask_annotations:
            # Return mask format
            # mask contains: original polygon if resample <= 0, resampled polygon if resample > 0
            result['mask'] = [mask_annotations[i]['polygon'] for i in range(len(mask_annotations))]
            
            # Create binary mask image
            if hasattr(image, 'shape'):
                # Tensor format
                image_size = (image.shape[-2], image.shape[-1])
            else:
                # PIL format
                image_size = image.size[::-1]  # (height, width)
            
            mask_image = self._create_mask_image(mask_annotations, image_size)
            result['mask_image'] = mask_image
            
            # Create skeleton image only if enabled (slow operation)
            if self.return_skeleton_image and all_mask_annotations_for_skeleton is not None:
                skeleton_image = self._create_skeleton_image(all_mask_annotations_for_skeleton, image_size, thickness=3)
                result['skeleton_image'] = skeleton_image
            
            # Create bbox from mask
            result['bbox'] = self._create_bbox_from_mask(mask_annotations, image_size)
            
            result['type'] = 'mask'
        elif bbox_annotations:
            # Return bbox format (no mask available)
            result['bbox'] = [bbox_annotations[i]['bbox'] for i in range(len(bbox_annotations))]  
            result['type'] = 'bbox'
        else:
            # This should never happen due to filtering, but handle gracefully
            print(f"Warning: No valid annotations found for {sample['base_name']}, retrying...")
            return self.__getitem__((idx + 1) % len(self))
        
        return result


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length annotations.
    Handles optional fields like 'mask_image' and 'visualize_image' gracefully.
    """
    # Get all possible keys from all samples
    all_keys = set()
    for sample in batch:
        all_keys.update(sample.keys())
    
    out = {}
    for key in all_keys:
        # Extract values from all samples (use get to handle missing keys)
        values = [s.get(key, None) for s in batch]
        
        # Skip if all values are None
        if all(v is None for v in values):
            continue
        
        # Check if all non-None values are tensors
        non_none_values = [v for v in values if v is not None]
        if all(isinstance(v, torch.Tensor) for v in non_none_values):
            # If all samples have this tensor field, stack them
            if len(non_none_values) == len(values):
                try:
                    out[key] = torch.stack(values, dim=0)
                except RuntimeError:
                    # Shape mismatch, keep as list
                    out[key] = values
            else:
                # Some samples don't have this field (e.g., mask_image for bbox-only samples)
                # Keep as list to maintain batch structure
                out[key] = values
        else:
            # Non-tensor fields (e.g., lists, strings), keep as list
            out[key] = values
    
    return out


def create_openkaryo_2d_dataloader(label_path: str,
                               image_path: str,
                               visualize_path: str,
                               transform: Optional[callable] = None,
                               batch_size: int = 1,
                               shuffle: bool = True,
                               num_workers: int = 4,
                               pin_memory: bool = True,
                               return_dataset: bool = False,
                               resample: int = 32,
                               data_augmentation: bool = True,
                               invert_colormap: bool = True,
                               return_skeleton_image: bool = False,
                               **kwargs) -> Union[DataLoader, OpenKaryo2DDataset]:
    """
    Create a DataLoader for OpenKaryo dataset with class-based sampling.
    
    Args:
        label_path (str): Path to label folder containing txt files
        image_path (str): Path to image folder
        visualize_path (str): Path to visualization folder
        transform (callable, optional): Transform to apply to images
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the dataset
        num_workers (int): Number of worker processes
        pin_memory (bool): Whether to pin memory
        return_dataset (bool): If True, return dataset instead of dataloader
        resample (int): Number of points to resample. Default 32.
            - If resample <= 0: mask returns original polygon
            - If resample > 0: mask returns N resampled points
        data_augmentation (bool): Whether to apply data augmentation (default: True)
        invert_colormap (bool): Whether to apply invert colormap (default: True)
        return_skeleton_image (bool): Whether to compute skeleton_image (default: False, slow)
        **kwargs: Additional arguments for DataLoader
    
    Returns:
        DataLoader or OpenKaryo2DDataset: Configured DataLoader instance or dataset
    """
    
    dataset = OpenKaryo2DDataset(
        label_path=label_path,
        image_path=image_path,
        visualize_path=visualize_path,
        transform=transform,
        resample=resample,
        data_augmentation=data_augmentation,
        invert_colormap=invert_colormap,
        return_skeleton_image=return_skeleton_image,
    )
    
    if return_dataset:
        return dataset
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn,
        **kwargs
    )


# Example usage
if __name__ == '__main__':
    from torchvision import transforms
    
    print("Testing YOLO format dataloader with class-based sampling...")
    # yolo_label_path = "./OpenKaryo/2D/Train/Label/YOLO/Mask/Banding"
    yolo_label_path = "./data/2D/train/label"
    yolo_image_path = "./data/2D/train/image"
    yolo_visualize_path = "./data/2D/train/visualization"
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        yolo_loader = create_openkaryo_2d_dataloader(
            label_path=yolo_label_path,
            image_path=yolo_image_path,
            visualize_path=yolo_visualize_path,
            transform=transform,
            batch_size=4,
            num_workers=2,
            resample=32,
            data_augmentation=True,  # Enable data augmentation
            invert_colormap=True,
            return_skeleton_image=True,  # Enable skeleton for testing visualization
        )
        
        print(f"Dataset size: {len(yolo_loader.dataset)}")
        print(f"Number of batches: {len(yolo_loader)}")
        
        for batch_idx, batch in enumerate(yolo_loader):
            print(f"\nBatch {batch_idx}:")
            print(f"Batch keys: {batch.keys()}")
            print(f"Image shape: {batch['image'].shape}")
            print(f"Class IDs: {batch['class_id']}")
            print(f"Types: {batch['type']}")
            print(f"Base names: {batch['base_name']}")
            
            # Show bbox examples
            bbox_indices = [i for i, t in enumerate(batch['type']) if t == 'bbox']
            if bbox_indices:
                print(f"BBox examples: {[batch['bbox'][i] for i in bbox_indices]}")
                print(f"BBox length: {len(batch['bbox'])}")
            
            # Show mask examples
            mask_indices = [i for i, t in enumerate(batch['type']) if t == 'mask']
            if mask_indices:
                print(f"Mask examples (first instance): {batch['mask'][mask_indices[0]][0][:10]}...")  # Show first 10 coords
                print(f"Mask polygon points count: {[len(batch['mask'][i]) // 2 for i in mask_indices]}")
                print(f"BBox examples: {[batch['bbox'][i] for i in mask_indices]}")
                print(f"BBox length: {[len(batch['bbox'][i]) for i in mask_indices]}")
                if 'mask_image' in batch and batch['mask_image'][0] is not None:
                    print(f"Mask image shape: {batch['mask_image'][0].shape}")
                if 'skeleton_image' in batch and batch['skeleton_image'][0] is not None:
                    print(f"Skeleton image shape: {batch['skeleton_image'][0].shape}")
                if 'psf_image' in batch:
                    print(f"PSF image shape: {batch['psf_image'].shape}")
                
            # save mask image and visualizations
            if mask_indices:
                import torchvision
                for i in mask_indices:
                    mask_image = batch['mask_image'][i]
                    skeleton_image = batch['skeleton_image'][i]
                    image = batch['image'][i]
                    psf_image = batch['psf_image'][i]
                    visualize_image = batch['visualize_image'][i]
                    shape_image = batch['shape_image'][i]
                    
                    # Ensure mask_image is a float tensor in [0,1] for save_image
                    if mask_image.dtype != torch.float32:
                        mask_image = mask_image.float()
                    # If mask_image is not in [0,1], normalize it
                    if mask_image.max() > 1.0 or mask_image.min() < 0.0:
                        mask_image = (mask_image - mask_image.min()) / (mask_image.max() - mask_image.min() + 1e-8)
                    # Add channel dimension if needed
                    if mask_image.dim() == 2:
                        mask_image = mask_image.unsqueeze(0).repeat(3, 1, 1)
                    
                    # Ensure skeleton_image is a float tensor in [0,1] for save_image
                    if skeleton_image.dtype != torch.float32:
                        skeleton_image = skeleton_image.float()
                    if skeleton_image.max() > 1.0 or skeleton_image.min() < 0.0:
                        skeleton_image = (skeleton_image - skeleton_image.min()) / (skeleton_image.max() - skeleton_image.min() + 1e-8)
                    if skeleton_image.dim() == 2:
                        skeleton_image = skeleton_image.unsqueeze(0).repeat(3, 1, 1)
                    
                    # make grid of images (including skeleton_image and psf_image)
                    grid = torchvision.utils.make_grid([mask_image, skeleton_image, image, psf_image, visualize_image, shape_image], nrow=6)
                    torchvision.utils.save_image(grid, f"{batch['base_name'][i]}.png")
            
            if batch_idx >= 2:  # Show only first 3 batches
                break
            
    except Exception as e:
        print(f"Loader error: {e}")
        import traceback
        traceback.print_exc()
