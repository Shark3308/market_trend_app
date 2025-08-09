"""
utils/filters.py - Data filtering and transformation utilities
Shared between retailer and consumer analytics pipelines
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np

class DataFilters:
    @staticmethod
    def filter_by_date(
        df: pd.DataFrame,
        date_col: str = 'date',
        days_back: int = 30,
        reference_date: datetime = None
    ) -> pd.DataFrame:
        """
        Filter DataFrame to only include recent records
        
        Args:
            df: Input DataFrame
            date_col: Name of datetime column
            days_back: Number of days to look back
            reference_date: Date to calculate from (default: now)
            
        Returns:
            Filtered DataFrame
        """
        cutoff = (reference_date or datetime.now()) - timedelta(days=days_back)
        return df[df[date_col] >= cutoff].copy()

    @staticmethod
    def detect_trending_products(
        transactions: pd.DataFrame,
        products: pd.DataFrame,
        current_date: datetime = None,
        short_window: int = 30,
        long_window: int = 90
    ) -> Dict[str, List]:
        """
        Identify products with accelerating salesa
        
        Args:
            transactions: DataFrame of transaction records
            products: DataFrame of product info
            current_date: Reference date for analysis
            short_window: Days for recent sales (default: 30)
            long_window: Days for baseline sales (default: 90)
            
        Returns:
            {
                'trending': List of trending product names,
                'declining': List of declining product names
            }
        """
        current_date = current_date or datetime.now()
        
        # Calculate sales velocity
        recent_sales = DataFilters.filter_by_date(
            transactions, 'date', short_window, current_date
        ).groupby('product_name')['units_sold'].sum()

        baseline_sales = DataFilters.filter_by_date(
            transactions, 'date', long_window, current_date - timedelta(days=short_window)
        ).groupby('product_name')['units_sold'].sum()

        # Calculate growth rates
        combined = pd.concat([
            recent_sales.rename('recent'),
            baseline_sales.rename('baseline')
        ], axis=1).fillna(0)

        combined['growth'] = (combined['recent'] - combined['baseline']) / (combined['baseline'] + 1)
        
        return {
            'trending': combined.nlargest(5, 'growth').index.tolist(),
            'declining': combined.nsmallest(5, 'growth').index.tolist()
        }

    @staticmethod
    def analyze_inventory(
        products: pd.DataFrame,
        sales_velocity: pd.Series,
        overstock_threshold: float = 3.0,
        understock_threshold: float = 0.5
    ) -> Dict[str, List]:
        """
        Identify inventory issues based on sales patterns
        
        Args:
            products: DataFrame with inventory data
            sales_velocity: Series of units sold per product
            overstock_threshold: Months of inventory considered overstocked
            understock_threshold: Months of inventory considered understocked
            
        Returns:
            {
                'overstocked': List of overstocked products,
                'understocked': List of understocked products
            }
        """
        merged = products.join(sales_velocity.rename('sales_30d'), how='left')
        merged['months_cover'] = merged['units_available'] / (merged['sales_30d'] / 30 + 0.1)  # +0.1 to avoid div/0
        
        return {
            'overstocked': merged[merged['months_cover'] > overstock_threshold].index.tolist(),
            'understocked': merged[merged['months_cover'] < understock_threshold].index.tolist()
        }

    @staticmethod
    def generate_price_recommendations(
        price_history: pd.DataFrame,
        current_prices: pd.DataFrame,
        current_date: datetime = None
    ) -> List[Dict]:
        """
        Generate consumer purchase recommendations based on price patterns
        
        Args:
            price_history: DataFrame with historical prices by month
            current_prices: DataFrame with current retailer prices
            current_date: Reference date for seasonal analysis
            
        Returns:
            List of recommendation dicts for each product
        """
        current_date = current_date or datetime.now()
        month = current_date.month
        
        # Melt price history into long format
        price_cols = [col for col in price_history.columns if col.startswith('Price_')]
        melted = price_history.melt(
            id_vars=['Product Name', 'Category'],
            value_vars=price_cols,
            var_name='Month',
            value_name='Price'
        )
        melted['Month'] = melted['Month'].str.replace('Price_', '')
        
        # Calculate price statistics
        stats = melted.groupby(['Product Name', 'Category'])['Price'].agg(
            ['min', 'max', 'mean', 'last']
        ).reset_index()
        
        # Merge with current prices
        merged = stats.merge(
            current_prices[['Product Name', 'Price']],
            on='Product Name',
            suffixes=('_historical', '_current')
        )
        
        recommendations = []
        for _, row in merged.iterrows():
            rec = {
                'product': row['Product Name'],
                'category': row['Category'],
                'current_price': row['Price_current'],
                'historical_low': row['min'],
                'historical_high': row['max'],
                'price_change_pct': round(
                    (row['Price_current'] - row['last']) / row['last'] * 100, 1
                )
            }
            
            # Recommendation logic
            if row['Price_current'] <= row['min']:
                rec['recommendation'] = 'Best Price - Buy Now'
                rec['confidence'] = 'high'
            elif month in [11, 12] and row['Category'] in ['Winter Wear', 'Heaters']:
                rec['recommendation'] = 'Seasonal High Demand'
                rec['confidence'] = 'medium'
            elif (row['Price_current'] - row['min']) / row['min'] < 0.1:
                rec['recommendation'] = 'Near Historical Low'
                rec['confidence'] = 'medium'
            else:
                rec['recommendation'] = 'Monitor Prices'
                rec['confidence'] = 'low'
            
            recommendations.append(rec)
        
        return recommendations

    @staticmethod
    def detect_seasonal_patterns(
        transactions: pd.DataFrame,
        min_seasonality: float = 1.5
    ) -> Dict[str, List[int]]:
        """
        Identify products with strong seasonal sales patterns
        
        Args:
            transactions: DataFrame of historical transactions
            min_seasonality: Minimum seasonal multiplier to consider
            
        Returns:
            Dict mapping product names to peak months
        """
        monthly = transactions.groupby([
            transactions['date'].dt.month,
            'product_name'
        ])['units_sold'].sum().unstack()
        
        seasonal = {}
        for product in monthly.columns:
            sales = monthly[product]
            peak = sales.max()
            if peak > (sales.mean() * min_seasonality):
                seasonal[product] = sales.nlargest(3).index.tolist()
        
        return seasonal