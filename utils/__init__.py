# app/__init__.py - Package initialization for Market Trend App backend

"""
Market Trend App Backend Modules

Provides:
- analyze_consumer: Consumer price trend analysis
- analyze_retailer: Retailer sales analytics
- utils: Shared utility functions
"""

__version__ = "1.0.0"
__all__ = []  # Explicit empty list - import modules directly

import logging
from pathlib import Path

# Configure package-level logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verify_data_paths():
    """Ensure required data directories exist"""
    base_dir = Path(__file__).parent.parent
    paths = [
        base_dir / 'data/raw',
        base_dir / 'data/processed'
    ]
    
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Verified path: {path}")

# Initialize when package is imported
verify_data_paths()
logger.info(f"Market Trend App backend v{__version__} initialized")

# Note: No direct imports here to prevent circular imports
# Modules should be imported directly where needed