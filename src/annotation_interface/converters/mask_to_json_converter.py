"""
Mask to JSON Converter for Annotations
======================================

Converts raster masks (PNG/TIFF) to JSON annotation format during dataset creation.
This pre-processes masks into an efficient JSON format for faster annotation loading.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


logger = logging.getLogger(__name__)


class MaskToJsonConverter:
    """
    Efficiently converts raster masks to JSON annotation format.
    
    Features:
    - Vectorized operations for speed
    - Batch processing support
    - Memory efficient streaming
    - Progress tracking
    - Error recovery
    """
    
    def __init__(self, ignore_index: int = 255, background_index: int = 0):
        """
        Initialize converter.
        
        Args:
            ignore_index: Pixel value to ignore (usually 255)
            background_index: Background pixel value (usually 0)
        """
        self.ignore_index = ignore_index
        self.background_index = background_index
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def convert_mask_to_json(
        self, 
        mask_path: Path, 
        output_path: Optional[Path] = None,
        dense_mask_path: Optional[Path] = None,
        iteration: int = 0
    ) -> Dict[str, Any]:
        """
        Convert a single mask file to JSON annotation format.
        
        Args:
            mask_path: Path to the mask file (PNG/TIFF)
            output_path: Optional path to save JSON (if None, returns dict only)
            dense_mask_path: Optional path to dense ground truth mask
            iteration: Iteration number for metadata
            
        Returns:
            Dictionary containing annotation data
        """
        try:
            # Load mask
            mask = np.array(Image.open(mask_path))
            
            # Extract annotation points
            points, metadata = self._extract_annotation_points(
                mask, 
                dense_mask_path,
                iteration
            )
            
            # Create annotation structure
            annotation = {
                'version': '2.0',
                'image_name': mask_path.stem,
                'image_path': str(mask_path),
                'width': mask.shape[1],
                'height': mask.shape[0],
                'annotations': {
                    'points': points,
                    'metadata': metadata
                },
                'stats': {
                    'total_points': len(points),
                    'classes_present': list(set(p[2] for p in points)) if points else [],
                    'conversion_timestamp': datetime.now().isoformat()
                }
            }
            
            # Save if output path provided
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(annotation, f, indent=2)
                self.logger.debug(f"Saved JSON annotation to {output_path}")
            
            return annotation
            
        except Exception as e:
            self.logger.error(f"Failed to convert {mask_path}: {e}")
            return {}
    
    def _extract_annotation_points(
        self, 
        mask: np.ndarray,
        dense_mask_path: Optional[Path] = None,
        iteration: int = 0
    ) -> Tuple[List[List[int]], Dict[str, Any]]:
        """
        Extract annotation points from mask using vectorized operations.
        
        Args:
            mask: Numpy array of mask
            dense_mask_path: Optional path to dense mask
            iteration: Iteration number
            
        Returns:
            Tuple of (points list, metadata dict)
        """
        # Find valid pixels (not background, not ignore)
        valid_mask = (mask != self.background_index) & (mask != self.ignore_index)
        
        if not np.any(valid_mask):
            return [], {}
        
        # Get coordinates and classes vectorized
        y_coords, x_coords = np.where(valid_mask)
        class_ids = mask[valid_mask]
        
        # Convert to 0-based if needed (some masks use 1-based indexing)
        if class_ids.min() > 0:
            class_ids = class_ids - 1
        
        # Create points list [[x, y, class_id], ...]
        points = np.column_stack([
            x_coords.astype(np.int32),
            y_coords.astype(np.int32),
            class_ids.astype(np.int32)
        ]).tolist()
        
        # Handle dense ground truth if provided
        dense_gt = []
        if dense_mask_path and dense_mask_path.exists():
            try:
                dense_mask = np.array(Image.open(dense_mask_path))
                dense_gt = dense_mask[valid_mask].astype(np.int32).tolist()
            except Exception as e:
                self.logger.warning(f"Could not load dense mask: {e}")
                dense_gt = class_ids.astype(np.int32).tolist()
        else:
            dense_gt = class_ids.astype(np.int32).tolist()
        
        # Create metadata
        metadata = {
            'dense_gt': dense_gt,
            'iteration_added': iteration,
            'timestamp': datetime.now().isoformat(),
            'total_points': len(points),
            'unique_classes': np.unique(class_ids).tolist()
        }
        
        return points, metadata
    
    def batch_convert_masks(
        self,
        mask_dir: Path,
        output_dir: Path,
        dense_mask_dir: Optional[Path] = None,
        max_workers: int = 4,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Convert all masks in a directory to JSON format in parallel.
        
        Args:
            mask_dir: Directory containing mask files
            output_dir: Directory to save JSON files
            dense_mask_dir: Optional directory with dense masks
            max_workers: Number of parallel workers
            show_progress: Show progress bar
            
        Returns:
            Summary statistics of conversion
        """
        # Find all mask files
        mask_files = list(mask_dir.glob("*.png")) + list(mask_dir.glob("*.tif"))
        
        if not mask_files:
            self.logger.warning(f"No mask files found in {mask_dir}")
            return {'converted': 0, 'failed': 0, 'total': 0}
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare tasks
        tasks = []
        for mask_path in mask_files:
            output_path = output_dir / f"{mask_path.stem}.json"
            dense_path = None
            if dense_mask_dir:
                dense_path = dense_mask_dir / mask_path.name
            tasks.append((mask_path, output_path, dense_path))
        
        # Process in parallel
        successful = 0
        failed = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.convert_mask_to_json, 
                    mask_path, 
                    output_path, 
                    dense_path
                ): mask_path 
                for mask_path, output_path, dense_path in tasks
            }
            
            # Progress bar setup
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Converting masks")
            
            for future in iterator:
                try:
                    result = future.result()
                    if result:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    self.logger.error(f"Task failed: {e}")
                    failed += 1
        
        # Summary
        summary = {
            'converted': successful,
            'failed': failed,
            'total': len(mask_files),
            'output_directory': str(output_dir),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save summary
        summary_path = output_dir / "_conversion_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Converted {successful}/{len(mask_files)} masks to JSON")
        return summary


def convert_dataset_masks_to_json(
    session_dir: Path,
    sparse_masks_only: bool = True,
    max_workers: int = 4
) -> bool:
    """
    Convenience function to convert all masks in a session to JSON.
    
    Args:
        session_dir: Session directory path
        sparse_masks_only: Only convert sparse masks (not dense)
        max_workers: Number of parallel workers
        
    Returns:
        True if successful
    """
    try:
        converter = MaskToJsonConverter()
        
        # Define paths
        dataset_dir = session_dir / "dataset"
        annotations_dir = session_dir / "iterations" / "iteration_0" / "annotations"
        
        # Convert training sparse masks
        if sparse_masks_only:
            train_sparse = dataset_dir / "train" / "sparse_masks"
            if train_sparse.exists():
                output_dir = annotations_dir / "sparse_annotations_json"
                result = converter.batch_convert_masks(
                    train_sparse,
                    output_dir,
                    dense_mask_dir=dataset_dir / "train" / "dense_masks",
                    max_workers=max_workers
                )
                logger.info(f"Training sparse masks conversion: {result}")
        
        # Convert validation masks if needed
        val_masks = dataset_dir / "val" / "masks"
        if val_masks.exists():
            output_dir = annotations_dir / "val_annotations_json"
            result = converter.batch_convert_masks(
                val_masks,
                output_dir,
                max_workers=max_workers
            )
            logger.info(f"Validation masks conversion: {result}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert dataset masks: {e}")
        return False