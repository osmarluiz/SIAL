"""
Annotation Tool Integration

Provides annotation tool launcher for SIAL active learning workflow.
Uses the SharedModulesAnnotationWidget from VIZ_SOFTWARE adapted for SIAL.
"""

from .launcher import launch_annotation_tool

__all__ = ['launch_annotation_tool']
