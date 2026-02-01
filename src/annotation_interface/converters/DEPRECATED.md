# DEPRECATED: Old Converter Modules

**Status:** ⚠️ DEPRECATED - Do not use
**Date:** 2025-01-25

## Notice

The converter modules in this directory are **DEPRECATED** and should **NOT** be used in new code.

### Deprecated Files

- ❌ `mask_to_json_converter.py` - Old converter (incompatible format)
- ❌ `annotation_format_utils.py` - Old format utilities (wrong format)
- ⚠️ `mask_generator.py` - May be used for mask generation only

### Why Deprecated?

These converters were extracted from VIZ_SOFTWARE but create **incompatible formats**:

1. **mask_to_json_converter.py** creates version "2.0" format:
   ```json
   {
     "version": "2.0",
     "annotations": {
       "points": [[x, y, class], ...]
     }
   }
   ```

2. **annotation_format_utils.py** uses parallel arrays format:
   ```json
   {
     "format_version": "1.0",
     "coordinates": [[x, y], ...],
     "class": [class, ...]
   }
   ```

### Production Format (What to Use Instead)

The **actual production VIZ_SOFTWARE format** is:
```json
{
  "format_version": "1.0",
  "image": {"name": "...", "width": ..., "height": ...},
  "annotations": [[x, y, class], [x, y, class], ...],
  "iteration": 0,
  "created_at": "2025-01-25T..."
}
```

## Replacement

### ✅ Use Instead: `src/session/simple_mask_converter.py`

This module creates the correct production format used in VIZ_SOFTWARE sessions.

**Import:**
```python
from src.session.simple_mask_converter import (
    convert_mask_to_json,
    json_to_mask,
    count_annotation_points,
    validate_annotation_format
)
```

**Benefits:**
- ✅ Uses production VIZ_SOFTWARE format
- ✅ 100% compatible with annotation tool
- ✅ Simple, efficient implementation
- ✅ Fully tested (21/21 tests passed)
- ✅ No hardcoded values (uses config)

### Example

**Before (WRONG):**
```python
from annotation_interface.converters.mask_to_json_converter import MaskToJsonConverter

converter = MaskToJsonConverter()
converter.convert_mask_to_json(
    mask_path=str(mask_path),
    output_json_path=str(json_path),  # Wrong parameter name
    image_name=...,                    # Doesn't exist
    width=...,                         # Doesn't exist
    height=...                         # Doesn't exist
)
# Creates wrong format (version 2.0)
```

**After (CORRECT):**
```python
from src.session.simple_mask_converter import convert_mask_to_json

convert_mask_to_json(
    mask_path=mask_path,
    output_path=json_path,
    ignore_index=6,  # From config
    iteration=0
)
# Creates correct production format (version 1.0)
```

## Migration Guide

If you have code using the old converters:

1. Replace imports:
   ```python
   # Old
   from annotation_interface.converters.mask_to_json_converter import MaskToJsonConverter

   # New
   from src.session.simple_mask_converter import convert_mask_to_json
   ```

2. Update API calls:
   ```python
   # Old
   converter = MaskToJsonConverter(ignore_index=6)
   result = converter.convert_mask_to_json(mask_path, output_path)

   # New
   success = convert_mask_to_json(mask_path, output_path, ignore_index=6, iteration=0)
   ```

3. Update validation:
   ```python
   # Old
   from annotation_interface.converters.annotation_format_utils import validate_annotation_format

   # New
   from src.session.simple_mask_converter import validate_annotation_format
   ```

## Why Keep These Files?

These files are kept for:
1. **Historical reference** - Understanding format evolution
2. **Documentation** - Learning what formats were tried
3. **Backwards compatibility** - Reading old format files (if needed)

But **DO NOT USE THEM** in new code.

## Future: Annotation Tool Integration

The annotation_interface directory will be updated in **Phase 6: Annotation Integration** to:
1. Use the correct production format
2. Update all imports to use simple_mask_converter
3. Refactor annotation_widget.py to work standalone

For now, use `src/session/simple_mask_converter.py` for all mask/JSON conversions.

## Questions?

See:
- [src/session/simple_mask_converter.py](../../src/session/simple_mask_converter.py) - Production converter
- [src/session/tests/README.md](../../src/session/tests/README.md) - Test documentation
- [TESTING_SUMMARY.md](../../TESTING_SUMMARY.md) - Test results
