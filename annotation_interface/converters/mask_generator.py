"""
Simple Mask Generator - JSON Annotation to Dense Mask Converter

Converts JSON point annotations to dense mask files for training data preparation.
Provides clean separation between annotation preprocessing and dataloader creation
following VIZ SOFTWARE principles of modularity and efficiency.

Responsibilities:
- Read JSON annotations from session iteration directories
- Extract ignore_index from dataset configuration
- Convert sparse point annotations to dense masks
- Save masks to dataset directories
- Validate conversion status and file integrity

Integration Points:
- Reads from session structure: annotations/points/iteration_X/*.json
- Uses dataset_metadata.json for configuration
- Outputs to: dataset/[split]/masks/*.png
- No dependencies on UI or training components

Architecture Notes:
- Processes one annotation at a time for memory efficiency
- Reads configuration once and caches for performance
- Atomic file operations for data integrity
- Comprehensive error handling with detailed logging
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
import numpy as np
from PIL import Image
import imageio

logger = logging.getLogger(__name__)


class ConversionStatus(Enum):
    """Status enumeration for mask conversion states."""
    NOT_GENERATED = "not_generated"
    READY = "ready"
    PARTIAL = "partial"
    GENERATING = "generating"
    ERROR = "error"


class MaskGenerator:
    """
    Core utility for converting JSON point annotations to dense mask files.

    Handles the complete pipeline from session JSON annotations to training-ready
    mask files, with proper error handling and status tracking.

    Attributes:
        session_path (Path): Path to session directory
        ignore_index (int): Pixel value for unlabeled areas
        dataset_metadata (dict): Cached dataset configuration
    """

    def __init__(self, session_path: Path):
        """
        Initialize mask generator with session path and load configuration.

        Args:
            session_path: Path to session directory containing config and annotations

        Raises:
            FileNotFoundError: If session config files are missing
            ValueError: If session configuration is invalid
        """
        self.session_path = Path(session_path)
        self.ignore_index = None
        self.dataset_metadata = None

        # Load and cache configuration
        self._load_session_configuration()

        logger.info(f"SimpleMaskGenerator initialized for session: {self.session_path}")

    def _load_session_configuration(self) -> None:
        """
        Load session configuration from dataset_metadata.json.

        Raises:
            FileNotFoundError: If dataset_metadata.json doesn't exist
            ValueError: If configuration format is invalid
        """
        metadata_path = self.session_path / 'config' / 'dataset_metadata.json'

        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")

        try:
            with open(metadata_path, 'r') as f:
                self.dataset_metadata = json.load(f)

            # Extract ignore_index from classes section (not class_info)
            class_info = self.dataset_metadata.get('class_info', {})
            classes_section = self.dataset_metadata.get('classes', {})
            self.ignore_index = classes_section.get('ignore_index', 255)

            logger.info(f"Loaded session config: ignore_index={self.ignore_index}")

        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid dataset metadata format: {e}")

    def get_ignore_index(self) -> int:
        """
        Get the ignore index value for unlabeled pixels.

        Returns:
            int: Ignore index value from dataset configuration
        """
        return self.ignore_index

    def convert_json_to_mask(self, json_path: Path, image_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Convert single JSON annotation file to dense mask array.

        Args:
            json_path: Path to JSON annotation file
            image_size: Optional (height, width) tuple, will be auto-detected if None

        Returns:
            np.ndarray: Dense mask array with point annotations placed

        Raises:
            FileNotFoundError: If JSON file doesn't exist
            ValueError: If JSON format is invalid or image size cannot be determined
        """
        if not json_path.exists():
            raise FileNotFoundError(f"JSON annotation not found: {json_path}")

        try:
            # Load JSON annotation
            with open(json_path, 'r') as f:
                annotation_data = json.load(f)

            # Get image dimensions
            if image_size is None:
                image_size = self._get_image_size_from_annotation(annotation_data, json_path)

            height, width = image_size

            # Create mask filled with ignore_index
            mask = np.full((height, width), self.ignore_index, dtype=np.uint8)

            # Place point annotations
            points = annotation_data.get('annotations', {}).get('points', [])

            for point in points:
                if len(point) >= 3:
                    x = int(point[0])
                    y = int(point[1])
                    class_id = int(point[2])

                    # Ensure coordinates are within bounds
                    if 0 <= x < width and 0 <= y < height:
                        mask[y, x] = class_id
                    else:
                        logger.warning(f"Point out of bounds: ({x}, {y}) for image size {image_size}")

            logger.debug(f"Converted {len(points)} points from {json_path.name}")
            return mask

        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid JSON annotation format in {json_path}: {e}")

    def _get_image_size_from_annotation(self, annotation_data: dict, json_path: Path) -> Tuple[int, int]:
        """
        Extract image size from annotation metadata or corresponding image file.

        Args:
            annotation_data: Loaded JSON annotation data
            json_path: Path to JSON file for context

        Returns:
            Tuple[int, int]: (height, width) of the image

        Raises:
            ValueError: If image size cannot be determined
        """
        # Try to get from JSON metadata
        if 'width' in annotation_data and 'height' in annotation_data:
            return (annotation_data['height'], annotation_data['width'])

        # Try to get from image file
        image_name = annotation_data.get('image_name')
        if image_name:
            # Look for corresponding image in dataset
            for split in ['train', 'val', 'test']:
                image_path = self.session_path / 'dataset' / split / 'images' / image_name

                # Try common extensions
                for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                    test_path = image_path.with_suffix(ext)
                    if test_path.exists():
                        try:
                            img = imageio.imread(test_path)
                            if img.ndim == 2:
                                return img.shape  # (height, width)
                            else:
                                return img.shape[:2]  # (height, width, channels)
                        except Exception as e:
                            logger.warning(f"Failed to read image {test_path}: {e}")
                            continue

        # Fallback to default size
        logger.warning(f"Could not determine image size for {json_path.name}, using default 512x512")
        return (512, 512)

    def save_mask(self, mask: np.ndarray, output_path: Path) -> bool:
        """
        Save mask array to PNG file with atomic operation.

        Args:
            mask: Dense mask array to save
            output_path: Path where mask should be saved

        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save with atomic operation (write to temp, then rename)
            temp_path = output_path.with_suffix('.tmp')

            # Convert to PIL Image and save
            mask_image = Image.fromarray(mask)
            mask_image.save(temp_path, 'PNG')

            # Atomic rename
            temp_path.rename(output_path)

            logger.debug(f"Saved mask to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save mask to {output_path}: {e}")

            # Cleanup temp file if it exists
            temp_path = output_path.with_suffix('.tmp')
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass

            return False

    def convert_iteration_annotations(self, iteration: int, overwrite: bool = False) -> Dict[str, Any]:
        """
        Convert all JSON annotations for a specific iteration to mask files.

        Args:
            iteration: Iteration number to process
            overwrite: Whether to overwrite existing mask files

        Returns:
            Dict with conversion results: {
                'status': ConversionStatus,
                'converted_count': int,
                'error_count': int,
                'skipped_count': int,
                'errors': List[str]
            }
        """
        result = {
            'status': ConversionStatus.GENERATING,
            'converted_count': 0,
            'error_count': 0,
            'skipped_count': 0,
            'errors': []
        }

        try:
            # Find annotation files for this iteration
            annotations_dir = self.session_path / 'annotations' / 'points' / f'iteration_{iteration}'

            if not annotations_dir.exists():
                result['status'] = ConversionStatus.NOT_GENERATED
                result['errors'].append(f"No annotations directory for iteration {iteration}")
                return result

            json_files = list(annotations_dir.glob('*.json'))

            if not json_files:
                result['status'] = ConversionStatus.NOT_GENERATED
                result['errors'].append(f"No JSON annotations found in iteration {iteration}")
                return result

            logger.info(f"Converting {len(json_files)} annotations for iteration {iteration}")

            # Process each JSON file
            for json_path in json_files:
                try:
                    # Determine output path
                    image_name = json_path.stem

                    # Find which split this image belongs to
                    output_path = self._find_mask_output_path(image_name)

                    if output_path is None:
                        result['error_count'] += 1
                        result['errors'].append(f"Could not determine output path for {image_name}")
                        continue

                    # Skip if exists and not overwriting
                    if output_path.exists() and not overwrite:
                        result['skipped_count'] += 1
                        continue

                    # Convert JSON to mask
                    mask = self.convert_json_to_mask(json_path)

                    # Save mask file
                    if self.save_mask(mask, output_path):
                        result['converted_count'] += 1
                    else:
                        result['error_count'] += 1
                        result['errors'].append(f"Failed to save mask for {image_name}")

                except Exception as e:
                    result['error_count'] += 1
                    result['errors'].append(f"Error processing {json_path.name}: {str(e)}")
                    logger.error(f"Error converting {json_path}: {e}")

            # Determine final status
            if result['error_count'] == 0:
                if result['converted_count'] > 0 or result['skipped_count'] > 0:
                    result['status'] = ConversionStatus.READY
                else:
                    result['status'] = ConversionStatus.NOT_GENERATED
            else:
                if result['converted_count'] > 0:
                    result['status'] = ConversionStatus.PARTIAL
                else:
                    result['status'] = ConversionStatus.ERROR

            logger.info(f"Conversion complete for iteration {iteration}: {result['converted_count']} converted, "
                       f"{result['error_count']} errors, {result['skipped_count']} skipped")

        except Exception as e:
            result['status'] = ConversionStatus.ERROR
            result['errors'].append(f"Conversion failed: {str(e)}")
            logger.error(f"Failed to convert iteration {iteration}: {e}")

        return result

    def _find_mask_output_path(self, image_name: str) -> Optional[Path]:
        """
        Find the correct output path for a mask based on image location.

        Args:
            image_name: Name of the image (without extension)

        Returns:
            Optional[Path]: Path where mask should be saved, None if not found
        """
        # Look for the image in each split to determine correct output location
        for split in ['train', 'val', 'test']:
            images_dir = self.session_path / 'dataset' / split / 'images'

            if images_dir.exists():
                # Check for image with common extensions
                for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                    image_path = images_dir / f"{image_name}{ext}"
                    if image_path.exists():
                        # Found the image, return corresponding mask path
                        mask_dir = self.session_path / 'dataset' / split / 'masks'
                        return mask_dir / f"{image_name}.png"

        return None

    def check_masks_exist(self, iteration: int) -> Dict[str, Any]:
        """
        Check the status of mask generation for a specific iteration.

        Args:
            iteration: Iteration number to check

        Returns:
            Dict with status information: {
                'status': ConversionStatus,
                'total_annotations': int,
                'existing_masks': int,
                'missing_masks': int,
                'details': Dict[str, int]  # per split
            }
        """
        result = {
            'status': ConversionStatus.NOT_GENERATED,
            'total_annotations': 0,
            'existing_masks': 0,
            'missing_masks': 0,
            'details': {}
        }

        try:
            # Count JSON annotations
            annotations_dir = self.session_path / 'annotations' / 'points' / f'iteration_{iteration}'

            if not annotations_dir.exists():
                return result

            json_files = list(annotations_dir.glob('*.json'))
            result['total_annotations'] = len(json_files)

            if result['total_annotations'] == 0:
                return result

            # Check for corresponding masks by finding where each image actually exists
            for json_file in json_files:
                image_name = json_file.stem
                found_mask = False

                # Find which split this image belongs to
                for split in ['train', 'val', 'test']:
                    # Check if image exists in this split
                    images_dir = self.session_path / 'dataset' / split / 'images'
                    image_found = False

                    if images_dir.exists():
                        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                            if (images_dir / f"{image_name}{ext}").exists():
                                image_found = True
                                break

                    if image_found:
                        # Image exists in this split, check for corresponding mask
                        masks_dir = self.session_path / 'dataset' / split / 'masks'
                        mask_path = masks_dir / f"{image_name}.png"

                        if split not in result['details']:
                            result['details'][split] = {'existing': 0, 'missing': 0}

                        if mask_path.exists():
                            result['details'][split]['existing'] += 1
                            result['existing_masks'] += 1
                            found_mask = True
                        else:
                            result['details'][split]['missing'] += 1
                            result['missing_masks'] += 1
                        break

                # Note: found_mask is handled in the loop above - no additional logic needed here

            # Determine status
            if result['existing_masks'] == 0:
                result['status'] = ConversionStatus.NOT_GENERATED
            elif result['missing_masks'] == 0:
                result['status'] = ConversionStatus.READY
            else:
                result['status'] = ConversionStatus.PARTIAL

            logger.debug(f"Mask status for iteration {iteration}: {result['existing_masks']}/{result['total_annotations']} exist")

        except Exception as e:
            result['status'] = ConversionStatus.ERROR
            logger.error(f"Failed to check mask status for iteration {iteration}: {e}")

        return result


# Utility functions for common operations
def validate_json_annotation(json_data: dict) -> bool:
    """
    Validate JSON annotation structure.

    Args:
        json_data: Loaded JSON annotation data

    Returns:
        bool: True if valid annotation format
    """
    try:
        # Check required fields
        if 'annotations' not in json_data:
            return False

        annotations = json_data['annotations']
        if 'points' not in annotations:
            return False

        # Validate points format
        points = annotations['points']
        if not isinstance(points, list):
            return False

        # Check each point has at least [x, y, class_id]
        for point in points:
            if not isinstance(point, list) or len(point) < 3:
                return False

        return True

    except Exception:
        return False


def create_output_directories(session_path: Path) -> None:
    """
    Ensure mask output directories exist for all splits.

    Args:
        session_path: Path to session directory
    """
    for split in ['train', 'val', 'test']:
        mask_dir = session_path / 'dataset' / split / 'masks'
        mask_dir.mkdir(parents=True, exist_ok=True)