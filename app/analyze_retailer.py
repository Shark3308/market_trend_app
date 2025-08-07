"""
analyze_retailer.py - Core retailer analytics engine
Processes: data/raw/retailer_products.csv, data/raw/transaction.csv
Outputs: data/processed/retailer_insights.json
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json

class RetailerAnalyticsEngine:
    def __init__(self):
        # Initialize with absolute path certainty
        self.project_root = Path(__file__).resolve().parent.parent
        self.raw_data_path = self.project_root / 'data' / 'raw'
        self.processed_data_path = self.project_root / 'data' / 'processed'
        
        # Validate paths exist
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data directory not found at {self.raw_data_path}")
        if not self.processed_data_path.exists():
            self.processed_data_path.mkdir(parents=True, exist_ok=True)

        # Load datasets with validation
        self.products_df = self._load_dataset('retailer_products.csv', [
            ('product_name', 'string'),
            ('category', 'string'),
            ('price', 'float64'),
            ('units_available', 'int32')
        ])
        
        self.transactions_df = self._load_dataset('transaction.csv', [
            ('product_name', 'string'),
            ('date', 'datetime64[ns]'),
            ('units_sold', 'int32'),
            ('price', 'float64')
        ]).rename(columns={'date': 'transaction_date'})

        self.current_date = datetime.now().date()

    def _load_dataset(self, filename: str, columns: list) -> pd.DataFrame:
        """Safe dataset loader with validation"""
        file_path = self.raw_data_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Required data file {filename} not found in {self.raw_data_path}")

        try:
            # Build dtype dictionary from specification
            dtypes = {col[0]: col[1] for col in columns}
            
            # Load with strict validation
            df = pd.read_csv(
                file_path,
                usecols=[col[0] for col in columns],
                dtype=dtypes,
                parse_dates=['date'] if filename == 'transaction.csv' else None,
                date_parser=lambda x: pd.to_datetime(x, errors='coerce') if filename == 'transaction.csv' else None
            )
            
            # Verify critical columns
            if filename == 'retailer_products.csv':
                if df['product_name'].isnull().any():
                    raise ValueError("Null values found in product_name - cannot proceed")
                df = df.set_index('product_name')
                
            return df.dropna()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {filename}: {str(e)}")

    def _get_sales_velocity(self, days: int = 30) -> pd.Series:
        """Calculate units sold per product in time window"""
        cutoff = self.current_date - timedelta(days=days)
        return (self.transactions_df[
            self.transactions_df['transaction_date'] >= cutoff
        ].groupby('product_name')['units_sold']
         .sum()
         .reindex(self.products_df.index, fill_value=0))

    def generate_insights(self) -> dict:
        """Master method to compile all analytics"""
        # Core metrics
        sales_30d = self._get_sales_velocity(30)
        sales_60d = self._get_sales_velocity(60)
        
        # Inventory analysis
        stock_ratio = (self.products_df['units_available'] / 
                      (sales_30d.replace(0, 1)))
        
        insights = {
            "metadata": {
                "analysis_date": self.current_date.isoformat(),
                "data_sources": {
                    "retailer_products": {
                        "rows": len(self.products_df),
                        "columns": list(self.products_df.columns)
                    },
                    "transactions": {
                        "rows": len(self.transactions_df),
                        "date_range": {
                            "start": self.transactions_df['transaction_date'].min().isoformat(),
                            "end": self.transactions_df['transaction_date'].max().isoformat()
                        }
                    }
                }
            },
            "products": {
                "total_active": len(self.products_df),
                "trending": (sales_30d - sales_60d)
                            .nlargest(5)
                            .index.tolist(),
                "declining": (sales_60d - sales_30d)
                            .nlargest(5)
                            .index.tolist()
            },
            "inventory": {
                "overstocked": self.products_df[stock_ratio > 3]
                              .index.tolist(),
                "understocked": self.products_df[stock_ratio < 0.5]
                                .index.tolist()
            },
            "price_analysis": {
                "highest_margin": self.products_df['price'].idxmax(),
                "best_value": (self.products_df['price'] / sales_30d)
                            .replace(float('inf'), 0)
                            .idxmin()
            }
        }

        # Write output with atomic write pattern
        output_file = self.processed_data_path / 'retailer_insights.json'
        temp_file = output_file.with_suffix('.tmp')
        
        try:
            with open(temp_file, 'w') as f:
                json.dump(insights, f, indent=2)
            temp_file.replace(output_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise RuntimeError(f"Failed to write insights: {str(e)}")
            
        return insights

if __name__ == "__main__":
    try:
        analyzer = RetailerAnalyticsEngine()
        print("Generating retailer insights...")
        results = analyzer.generate_insights()
        print(f"Success! Results saved to {analyzer.processed_data_path/'retailer_insights.json'}")
        print(f"Top trending product: {results['products']['trending'][0]}")
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        exit(1)