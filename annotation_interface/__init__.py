"""
Annotation Interface - Professional PyQt5 Annotation System

Extracted from VIZ_SOFTWARE for use in the active learning workflow.

Key Modules:
- converters: ⚠️ DEPRECATED - Use src.session.simple_mask_converter instead
- core: Base protocols and interfaces
- canvas: Image display and interaction
- tools: Point annotation tools
- overlays: Prediction and ground truth overlays
- io: JSON loading/saving with file locking
- data: Annotation data storage
- navigation: Image navigation controls
- display: Channel mapping utilities
- ui: UI components and layouts
- controls: Control panels
- widgets: Main annotation widget (requires Phase 6 refactoring)

⚠️ DEPRECATION WARNING:
The converter modules are DEPRECATED. They create incompatible formats.

DO NOT USE:
- annotation_interface.converters.mask_to_json_converter
- annotation_interface.converters.annotation_format_utils

USE INSTEAD:
- src.session.simple_mask_converter (production format)

See: annotation_interface/converters/DEPRECATED.md for details

Note: The annotation_widget.py requires refactoring to work outside VIZ_SOFTWARE.
This will be completed in Phase 6: Annotation Integration.
"""

__version__ = "0.1.0"

# DEPRECATED: Do not use these converters in new code
# See converters/DEPRECATED.md for migration guide
#
# Use src.session.simple_mask_converter instead:
#   from src.session.simple_mask_converter import (
#       convert_mask_to_json,
#       json_to_mask,
#       validate_annotation_format
#   )
#
# Keeping these imports for backwards compatibility only
import warnings

warnings.warn(
    "annotation_interface.converters modules are DEPRECATED. "
    "Use src.session.simple_mask_converter instead. "
    "See annotation_interface/converters/DEPRECATED.md for details.",
    DeprecationWarning,
    stacklevel=2
)

from .converters.annotation_format_utils import (
    validate_annotation_format,
    convert_from_old_format,
    create_annotation_data
)

from .converters.mask_generator import MaskGenerator
from .converters.mask_to_json_converter import MaskToJsonConverter

__all__ = [
    'validate_annotation_format',
    'convert_from_old_format',
    'create_annotation_data',
    'MaskGenerator',
    'MaskToJsonConverter',
]
