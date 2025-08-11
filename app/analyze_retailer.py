"""
analyze_retailer.py - Enhanced Retailer Analytics Engine
Processes: data/raw/retailer_products.csv, data/raw/transaction.csv, data/raw/prices.csv
Outputs: data/processed/retailer_insights.json
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import numpy as np

class RetailerAnalyticsEngine:
    def __init__(self):
        # Initialize paths
        self.project_root = Path(__file__).resolve().parent.parent
        self.raw_data_path = self.project_root / 'data' / 'raw'
        self.processed_data_path = self.project_root / 'data' / 'processed'
        
        # Validate paths
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data directory not found at {self.raw_data_path}")
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

        # Load datasets
        self._load_datasets()
        self.current_date = datetime.now().date()

    def _load_datasets(self):
        """Load and validate all required datasets"""
        try:
            # Load products data
            self.products_df = pd.read_csv(
                self.raw_data_path / 'retailer_products.csv',
                usecols=['product_name', 'category', 'price', 'units_available'],
                dtype={
                    'product_name': 'string',
                    'category': 'string',
                    'price': 'float64',
                    'units_available': 'int32'
                }
            ).set_index('product_name').dropna()

            # Load transactions data
            self.transactions_df = pd.read_csv(
                self.raw_data_path / 'transaction.csv',
                parse_dates=['date_sold'],
                dtype={
                    'product_name': 'string',
                    'units_sold': 'int32',
                    'price': 'float64'
                }
            ).rename(columns={'date_sold': 'transaction_date'}).dropna()

            # Load prices data
            self.prices_df = pd.read_csv(
                self.raw_data_path / 'prices.csv',
                dtype={
                    'Product Name': 'string',
                    'Category': 'string',
                    **{f'Price_{month}': 'float64' for month in [
                        'January', 'February', 'March', 'April',
                        'May', 'June', 'July', 'August',
                        'September', 'October', 'November', 'December'
                    ]}
                }
            ).dropna()

        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}")

    def _process_monthly_data(self):
        """Process monthly price and sales data"""
        # Melt prices data to long format
        price_cols = [col for col in self.prices_df.columns if col.startswith('Price_')]
        prices_long = self.prices_df.melt(
            id_vars=['Product Name', 'Category'],
            value_vars=price_cols,
            var_name='month',
            value_name='price'
        )
        prices_long['month'] = prices_long['month'].str.replace('Price_', '')
        
        # Add month to transactions
        transactions = self.transactions_df.copy()
        transactions['month'] = transactions['transaction_date'].dt.strftime('%B')
        transactions['year'] = transactions['transaction_date'].dt.year
        
        return prices_long, transactions

    def _get_sales_trends(self, window_days: int = 30):
        """Calculate sales trends with moving averages"""
        cutoff = self.current_date - timedelta(days=window_days)
        recent_sales = self.transactions_df[
            self.transactions_df['transaction_date'] >= cutoff
        ]
        
        # Calculate daily sales
        daily_sales = recent_sales.groupby([
            'product_name',
            pd.Grouper(key='transaction_date', freq='D')
        ])['units_sold'].sum().unstack().fillna(0)
        
        # 7-day moving average
        moving_avg = daily_sales.rolling(window=7, axis=1).mean()
        
        return {
            'total_sold': daily_sales.sum(axis=1),
            'trend_slope': moving_avg.apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0], 
                axis=1
            )
        }

    def generate_insights(self) -> dict:
        """Generate comprehensive retailer insights"""
        try:
            # Process monthly data
            prices_long, transactions = self._process_monthly_data()
            
            # Calculate sales trends
            sales_trends = self._get_sales_trends()
            
            # Inventory analysis
            stock_ratio = (
                self.products_df['units_available'] /
                sales_trends['total_sold'].replace(0, 1)
            )
            # Price analysis
            current_prices = self.products_df['price']
            price_stats = prices_long.groupby('Product Name')['price'].agg([
                'min', 'max', 'mean'
            ])
            
            # Compile insights
            insights = {
                "metadata": {
                    "analysis_date": self.current_date.isoformat(),
                    "data_range": {
                        "transactions": {
                            "start": self.transactions_df['transaction_date'].min().isoformat(),
                            "end": self.transactions_df['transaction_date'].max().isoformat()
                        },
                        "prices": {
                            "months_available": list(prices_long['month'].unique())
                        }
                    }
                },
                "product_analytics": {
                    "trending": sales_trends['total_sold']
                                .nlargest(5)
                                .index.tolist(),
                    "declining": sales_trends['total_sold']
                                .nsmallest(5)
                                .index.tolist(),
                    "price_sensitivity": price_stats.join(current_prices)
                                .assign(price_change=lambda x: (x['price'] - x['mean']) / x['mean'])
                                .sort_values('price_change')
                                .head(5)
                                .index.tolist()
                },
                "inventory_recommendations": {
                    "overstocked": stock_ratio[stock_ratio > 3]
                                 .sort_values(ascending=False)
                                 .index.tolist(),
                    "understocked": stock_ratio[stock_ratio < 0.5]
                                   .sort_values()
                                   .index.tolist()
                },
                "monthly_insights": {
                    "best_selling_months": transactions.groupby('month')['units_sold']
                                      .sum()
                                      .nlargest(3)
                                      .index.tolist(),
                    "highest_margin_months": transactions.groupby('month')
                                        .apply(lambda x: (x['price'] * x['units_sold']).sum() / x['units_sold'].sum())
                                        .nlargest(3)
                                        .index.tolist()
                }
            }

            # Save results
            output_file = self.processed_data_path / 'retailer_insights.json'
            with open(output_file, 'w') as f:
                json.dump(insights, f, indent=2)
                
            return insights
            
        except Exception as e:
            raise RuntimeError(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    try:
        analyzer = RetailerAnalyticsEngine()
        print("Generating retailer insights...")
        results = analyzer.generate_insights()
        print(f"Success! Results saved to {analyzer.processed_data_path/'retailer_insights.json'}")
        print(f"Top trending product: {results['product_analytics']['trending'][0]}")
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        exit(1)