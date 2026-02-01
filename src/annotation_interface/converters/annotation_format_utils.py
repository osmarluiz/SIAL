"""
Utility functions for handling the new annotation format.
Provides validation, migration, and helper functions for the parallel array format.
"""

import json
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
from pathlib import Path


def validate_annotation_format(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate annotation data in the new parallel array format.
    
    Args:
        data: Annotation data dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check required fields
        required_fields = ["format_version", "image", "coordinates", "class"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"
        
        # Check array consistency
        n_coordinates = len(data["coordinates"])
        n_classes = len(data["class"])
        
        if n_coordinates != n_classes:
            return False, f"Array length mismatch: coordinates={n_coordinates}, classes={n_classes}"
        
        # Check optional arrays
        for optional_field in ["prediction", "gt", "confidence"]:
            if optional_field in data and data[optional_field] is not None:
                if len(data[optional_field]) != n_coordinates:
                    return False, f"Optional array {optional_field} length mismatch: expected {n_coordinates}, got {len(data[optional_field])}"
        
        # Validate stats if present
        if "stats" in data:
            stats = data["stats"]
            if "total_points" in stats and stats["total_points"] != n_coordinates:
                return False, f"Stats total_points mismatch: expected {n_coordinates}, got {stats['total_points']}"
        
        # Validate image bounds if coordinates exist
        if n_coordinates > 0 and "image" in data:
            width = data["image"].get("width", 0)
            height = data["image"].get("height", 0)
            
            for i, (x, y) in enumerate(data["coordinates"]):
                if not (0 <= x < width and 0 <= y < height):
                    return False, f"Point {i} at ({x}, {y}) is outside image bounds ({width}x{height})"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def convert_from_old_format(old_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert from old [x, y, class] points format to new parallel arrays format.
    
    Args:
        old_data: Data in old format
        
    Returns:
        Data in new format
    """
    try:
        # Extract points from old format
        if "annotations" in old_data and "points" in old_data["annotations"]:
            points = old_data["annotations"]["points"]
        elif "points" in old_data:
            points = old_data["points"]
        else:
            points = []
        
        # Convert to parallel arrays
        if points:
            coordinates = [[point[0], point[1]] for point in points]
            classes = [point[2] for point in points]
        else:
            coordinates = []
            classes = []
        
        # Calculate stats
        total_points = len(coordinates)
        class_counts = dict(Counter(classes)) if classes else {}
        unique_classes = sorted(set(classes)) if classes else []
        
        # Get image info
        width = old_data.get("width", 512)
        height = old_data.get("height", 512)
        image_name = old_data.get("image_name", "unknown.png")
        
        sparsity_ratio = total_points / (width * height) if width * height > 0 else 0
        
        # Create new format
        new_data = {
            "format_version": "1.0",
            "image": {
                "name": image_name,
                "width": width,
                "height": height
            },
            "coordinates": coordinates,
            "class": classes,
            "stats": {
                "total_points": total_points,
                "sparsity_ratio": sparsity_ratio,
                "class_counts": {str(k): v for k, v in class_counts.items()},
                "unique_classes": unique_classes
            },
            "provenance": {
                "source": "migrated_from_old_format",
                "method": "format_conversion",
                "iteration": old_data.get("annotations", {}).get("metadata", {}).get("iteration_added", 0)
            }
        }
        
        return new_data
        
    except Exception as e:
        raise ValueError(f"Failed to convert from old format: {str(e)}")


def convert_to_internal_format(data: Dict[str, Any]) -> List[List[int]]:
    """
    Convert from parallel arrays format to internal [x, y, class] format.
    
    Args:
        data: Data in new parallel arrays format
        
    Returns:
        List of [x, y, class] points for internal use
    """
    coordinates = data["coordinates"]
    classes = data["class"]
    
    return [[coord[0], coord[1], cls] for coord, cls in zip(coordinates, classes)]


def create_annotation_data(coordinates: List[List[int]], 
                          classes: List[int], 
                          image_name: str,
                          width: int, 
                          height: int,
                          **optional) -> Dict[str, Any]:
    """
    Create annotation data in the new format.
    
    Args:
        coordinates: List of [x, y] coordinates
        classes: List of class IDs
        image_name: Name of the image
        width: Image width
        height: Image height
        **optional: Optional arrays (prediction, gt, confidence)
        
    Returns:
        Annotation data in new format
    """
    import datetime
    from collections import Counter
    
    # Calculate stats
    total_points = len(coordinates)
    class_counts = dict(Counter(classes)) if classes else {}
    unique_classes = sorted(set(classes)) if classes else []
    sparsity_ratio = total_points / (width * height) if width * height > 0 else 0
    
    # Base data
    data = {
        "format_version": "1.0",
        "image": {
            "name": image_name,
            "width": width,
            "height": height
        },
        "coordinates": coordinates,
        "class": classes,
        "stats": {
            "total_points": total_points,
            "sparsity_ratio": sparsity_ratio,
            "class_counts": {str(k): v for k, v in class_counts.items()},
            "unique_classes": unique_classes
        },
        "provenance": {
            "source": optional.get("source", "manual_annotation"),
            "method": optional.get("method", "interactive_annotation_tool"),
            "created_at": datetime.datetime.now().isoformat(),
            "iteration": optional.get("iteration", 0)
        }
    }
    
    # Add optional arrays
    for field_name in ["prediction", "gt", "confidence"]:
        if field_name in optional and optional[field_name] is not None:
            data[field_name] = optional[field_name]
    
    return data


def get_format_info(data: Dict[str, Any]) -> str:
    """
    Get human-readable information about the annotation format.
    
    Args:
        data: Annotation data
        
    Returns:
        Format information string
    """
    if "format_version" in data:
        stats = data.get("stats", {})
        return (f"New format v{data['format_version']}: "
                f"{stats.get('total_points', 0)} points, "
                f"{len(stats.get('unique_classes', []))} classes, "
                f"sparsity {stats.get('sparsity_ratio', 0):.6f}")
    elif "annotations" in data:
        points = data["annotations"].get("points", [])
        return f"Old format: {len(points)} points"
    else:
        return "Unknown format"


def migrate_annotation_file(file_path: Path, backup: bool = True) -> bool:
    """
    Migrate an annotation file from old to new format.
    
    Args:
        file_path: Path to annotation file
        backup: Whether to create backup of old file
        
    Returns:
        True if migration successful
    """
    try:
        # Load current data
        with open(file_path, 'r') as f:
            old_data = json.load(f)
        
        # Check if already in new format
        if "format_version" in old_data:
            print(f"File {file_path} already in new format")
            return True
        
        # Create backup if requested
        if backup:
            backup_path = file_path.with_suffix('.json.backup')
            with open(backup_path, 'w') as f:
                json.dump(old_data, f, indent=2)
            print(f"Created backup: {backup_path}")
        
        # Convert to new format
        new_data = convert_from_old_format(old_data)
        
        # Validate
        is_valid, error = validate_annotation_format(new_data)
        if not is_valid:
            raise ValueError(f"Converted data is invalid: {error}")
        
        # Save new format
        with open(file_path, 'w') as f:
            json.dump(new_data, f, indent=2)
        
        print(f"Migrated {file_path}: {get_format_info(old_data)} -> {get_format_info(new_data)}")
        return True
        
    except Exception as e:
        print(f"Failed to migrate {file_path}: {e}")
        return False


if __name__ == "__main__":
    # Test the utilities
    print("Testing annotation format utilities...")
    
    # Test data
    test_coordinates = [[100, 200], [150, 250], [200, 300]]
    test_classes = [1, 2, 1]
    
    # Create data
    data = create_annotation_data(
        test_coordinates, test_classes, 
        "test.png", 512, 512,
        source="test"
    )
    
    # Validate
    is_valid, msg = validate_annotation_format(data)
    print(f"Validation: {is_valid} - {msg}")
    
    # Convert to internal
    internal = convert_to_internal_format(data)
    print(f"Internal format: {internal}")
    
    # Get info
    info = get_format_info(data)
    print(f"Format info: {info}")