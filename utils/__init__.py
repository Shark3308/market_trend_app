# utils/__init__.py - Initialization for shared utilities

"""
Market Trend App - Utility Module

Provides shared functionality for:
- CSV loading and validation (csv_loader.py)
- Data filtering operations (filters.py)
"""

__version__ = "1.0.0"
__all__ = ['csv_loader', 'filters']  # Explicitly expose public modules

import logging
from pathlib import Path

# Set up utility-specific logging
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())  # Default no output

def get_data_path(file_type: str) -> Path:
    """Get absolute path to data files based on type ('raw' or 'processed')"""
    base_dir = Path(__file__).parent.parent
    if file_type == 'raw':
        return base_dir / 'data/raw'
    elif file_type == 'processed':
        return base_dir / 'data/processed'
    raise ValueError(f"Unknown file_type: {file_type}. Use 'raw' or 'processed'")

# Package-level initialization
_logger.info(f"Market Trend Utils v{__version__} initialized")