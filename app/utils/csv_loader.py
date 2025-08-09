"""
utils/csv_loader.py - Centralized CSV loading and validation
Handles all raw data loading for both retailer and consumer analytics
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Union
import numpy as np

class CSVLoader:
    # Define expected schemas for each CSV file
    SCHEMAS = {
        'retailer_products.csv': {
            'columns': [
                ('product_name', 'string'),
                ('category', 'string'),
                ('price', 'float64'),
                ('units_available', 'int32')
            ],
            'required': True,
            'index_col': 'product_name'
        },
        'transaction.csv': {
            'columns': [
                ('product_name', 'string'),
                ('date', 'datetime64[ns]'),
                ('units_sold', 'int32'),
                ('price', 'float64')
            ],
            'required': True
        },
        'prices.csv': {
            'columns': [
                ('Product Name', 'string'),
                ('Category', 'string'),
                *[(f'Price_{month}', 'float64') for month in [
                    'January', 'February', 'March', 'April',
                    'May', 'June', 'July', 'August',
                    'September', 'October', 'November', 'December'
                ]]
            ],
            'required': False
        }
    }

    def __init__(self, base_dir: Union[str, Path] = None):
        """
        Initialize with project directory
        Args:
            base_dir: Project root path (default: two levels up from this file)
        """
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.raw_data_path = self.base_dir / 'data' / 'raw'
        self._validate_paths()

    def _validate_paths(self):
        """Ensure required directories exist"""
        if not self.raw_data_path.exists():
            raise FileNotFoundError(
                f"Raw data directory not found at {self.raw_data_path}"
            )
        self.raw_data_path.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, filename: str) -> pd.DataFrame:
        """
        Load and validate a dataset
        Args:
            filename: Name of CSV file in raw data directory
        Returns:
            Cleaned DataFrame with validated schema
        """
        if filename not in self.SCHEMAS:
            raise ValueError(f"No schema defined for {filename}")

        schema = self.SCHEMAS[filename]
        file_path = self.raw_data_path / filename

        if not file_path.exists():
            if schema['required']:
                raise FileNotFoundError(f"Required file {filename} not found")
            return pd.DataFrame()

        try:
            # Build dtype and parse_dates config
            dtype = {col[0]: col[1] for col in schema['columns']}
            parse_dates = [
                col[0] for col in schema['columns'] 
                if col[1] == 'datetime64[ns]'
            ]

            # Load with validation
            df = pd.read_csv(
                file_path,
                usecols=[col[0] for col in schema['columns']],
                dtype=dtype,
                parse_dates=parse_dates,
                date_parser=lambda x: pd.to_datetime(x, errors='coerce')
            )

            # Post-load validation
            if df.empty:
                raise ValueError(f"{filename} is empty after loading")

            # Check for nulls in string columns
            string_cols = [col[0] for col in schema['columns'] if col[1] == 'string']
            for col in string_cols:
                if df[col].isnull().any():
                    raise ValueError(f"Null values found in required column {col}")

            # Set index if specified
            if 'index_col' in schema:
                df = df.set_index(schema['index_col'])

            return df

        except Exception as e:
            raise RuntimeError(f"Failed to load {filename}: {str(e)}")

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all datasets defined in SCHEMAS"""
        return {
            name: self.load_dataset(name)
            for name in self.SCHEMAS.keys()
        }

# Helper functions for direct usage
def load_retailer_data() -> pd.DataFrame:
    """Convenience function for retailer products data"""
    return CSVLoader().load_dataset('retailer_products.csv')

def load_transaction_data() -> pd.DataFrame:
    """Convenience function for transaction data"""
    return CSVLoader().load_dataset('transaction.csv')

def load_price_history() -> pd.DataFrame:
    """Convenience function for price history data"""
    return CSVLoader().load_dataset('prices.csv')