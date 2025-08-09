# analyze_consumer.py - Market Trend Analyzer (Consumer Side)
# Processes: prices.csv, transactions.csv, retailer_products.csv 
# 

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from scipy.stats import linregress

class ConsumerPriceAnalyzer:
    def __init__(self):
        """Initialize with data paths and current analysis period"""
        self.base_dir = Path(__file__).parent.parent
        self.raw_path = self.base_dir / 'data' / 'raw'
        self.processed_path = self.base_dir / 'data' / 'processed'
        
        # Validate paths
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data not found at {self.raw_path}")
        self.processed_path.mkdir(exist_ok=True)
        
        # Current analysis parameters
        self.current_date = datetime.now()
        self.trend_window = 90  # Days to analyze trends
        
        # Load all required datasets
        self._load_datasets()

    def _load_datasets(self):
        """Load and validate all input CSVs"""
        try:
            # Load with explicit column handling
            self.products = pd.read_csv(
                self.raw_path / 'retailer_products.csv',
                usecols=['Product Name', 'Category', 'Price', 'Units Sold', 'Units Available']
            ).dropna()
            
            self.prices = pd.read_csv(
                self.raw_path / 'prices.csv',
                usecols=lambda c: c.startswith('Product') or c.startswith('Category') or c.startswith('Price_')
            ).dropna()
            
            self.transactions = pd.read_csv(
                self.raw_path / 'transactions.csv',
                parse_dates=['Date'],
                usecols=['Product Name', 'Date', 'Units Sold', 'Price']
            ).dropna()
            
            # Convert price columns to numeric
            price_cols = [c for c in self.prices.columns if c.startswith('Price_')]
            self.prices[price_cols] = self.prices[price_cols].apply(pd.to_numeric, errors='coerce')
            
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}")

    def _analyze_trends(self):
        """Core trend analysis logic"""
        # 1. Price Trends
        price_cols = [c for c in self.prices.columns if c.startswith('Price_')]
        price_stats = self.prices.melt(
            id_vars=['Product Name', 'Category'],
            value_vars=price_cols,
            var_name='Month',
            value_name='Price'
        )
        price_stats['Month'] = price_stats['Month'].str.replace('Price_', '')
        
        # 2. Demand Trends (last 90 days)
        cutoff_date = self.current_date - timedelta(days=self.trend_window)
        recent_trans = self.transactions[self.transactions['Date'] >= cutoff_date]
        
        # 3. Merge all insights
        merged = (
            self.products[['Product Name', 'Category', 'Price']]
            .merge(
                price_stats.groupby(['Product Name', 'Category'])['Price']
                .agg(['min', 'max', 'mean', 'last'])
                .reset_index(),
                on=['Product Name', 'Category']
            )
            .merge(
                recent_trans.groupby('Product Name')['Units Sold']
                .sum()
                .rename('recent_sales'),
                left_on='Product Name',
                right_index=True
            )
        )
        
        # Calculate metrics
        merged['price_change'] = (merged['Price'] - merged['last']) / merged['last'] * 100
        merged['discount_pct'] = (merged['max'] - merged['Price']) / merged['max'] * 100
        merged['value_score'] = merged['recent_sales'] / merged['Price']
        
        return merged

    def _generate_recommendations(self, trends_df):
        """Generate data-driven purchase recommendations"""
        recommendations = []
        
        for _, row in trends_df.iterrows():
            # Calculate demand trend slope
            product_trans = self.transactions[
                (self.transactions['Product Name'] == row['Product Name']) & 
                (self.transactions['Date'] >= self.current_date - timedelta(days=30))
            ]
            if len(product_trans) > 1:
                x = (product_trans['Date'] - product_trans['Date'].min()).dt.days
                slope = linregress(x, product_trans['Units Sold']).slope
            else:
                slope = 0
            
            # Recommendation logic
            if row['price_change'] < -5 and slope < 0:
                rec = "Strong Wait - prices and demand falling"
            elif row['discount_pct'] > 20:
                rec = "Hot Deal - significant discount"
            elif row['Price'] <= row['min']:
                rec = "Lowest Price - buy now"
            elif slope > 2:
                rec = "Trending Up - buy before price increases"
            else:
                rec = "Neutral - monitor prices"
                
            recommendations.append({
                'product': row['Product Name'],
                'category': row['Category'],
                'price': round(row['Price'], 2),
                'price_change_pct': round(row['price_change'], 1),
                'discount_pct': round(row['discount_pct'], 1),
                'demand_trend': round(slope, 2),
                'recommendation': rec
            })
        
        return sorted(recommendations, key=lambda x: abs(x['discount_pct']), reverse=True)

    def generate_insights(self):
        """Generate complete consumer insights package"""
        try:
            trends = self._analyze_trends()
            recommendations = self._generate_recommendations(trends)
            
            # Compile final output
            insights = {
                'analysis_date': self.current_date.strftime('%Y-%m-%d'),
                'trend_window_days': self.trend_window,
                'top_recommendations': recommendations[:10],
                'best_value_products': (
                    trends.nlargest(5, 'value_score')
                    [['Product Name', 'Category', 'Price', 'value_score']]
                    .rename(columns={'value_score': 'sales_per_dollar'})
                    .to_dict('records')
                ),
                'seasonal_trends': self._get_seasonal_trends()
            }
            
            # Save to JSON
            output_file = self.processed_path / 'consumer_insights.json'
            with open(output_file, 'w') as f:
                json.dump(insights, f, indent=2)
                
            return insights
            
        except Exception as e:
            raise RuntimeError(f"Analysis failed: {str(e)}")

    def _get_seasonal_trends(self):
        """Identify seasonal sales patterns"""
        monthly_sales = (
            self.transactions.groupby([
                'Product Name',
                self.transactions['Date'].dt.month
            ])['Units Sold'].sum().unstack()
        )
        
        seasonal = []
        for product, sales in monthly_sales.iterrows():
            avg = sales.mean()
            peak_months = sales[sales > 2 * avg].index.tolist()
            if peak_months:
                seasonal.append({
                    'product': product,
                    'peak_months': peak_months,
                    'seasonality_ratio': round(sales.max() / avg, 2)
                })
        
        return seasonal

if __name__ == "__main__":
    try:
        analyzer = ConsumerPriceAnalyzer()
        print("Generating consumer insights...")
        results = analyzer.generate_insights()
        print(f"Success! Saved to {analyzer.processed_path/'consumer_insights.json'}")
        print("Top recommendation:", results['top_recommendations'][0])
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)