#!/usr/bin/env python3
"""
Demand analysis module for market_trend_app.
Analyzes product demand patterns for retailer and consumer contexts.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
from utils.csv_loader import CSVLoader
from utils.filters import DataFilters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DemandAnalyzer:
    """Analyzes product demand patterns from transaction and product data."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize DemandAnalyzer with data directory.
        
        Args:
            data_dir: Directory containing input CSV files
        """
        self.data_dir = Path(data_dir)
        self.insights = {
            "metadata": {},
            "trending_products": [],
            "declining_products": [],
            "seasonal_products": [],
            "sales_velocity_changes": []
        }
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load required datasets using CSVLoader.
        
        Returns:
            Tuple of (transactions, retailer_products, prices) DataFrames
            
        Raises:
            FileNotFoundError: If any required file is missing
        """
        try:
            loader = CSVLoader(self.data_dir)
            transactions = loader.load_csv("transactions.csv")
            retailer_products = loader.load_csv("retailer_products.csv")
            prices = loader.load_csv("prices.csv")
            
            logger.info("Successfully loaded all datasets")
            return transactions, retailer_products, prices
            
        except FileNotFoundError as e:
            logger.error(f"Missing required data file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, transactions: pd.DataFrame, 
                       retailer_products: pd.DataFrame,
                       prices: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and merge datasets for analysis.
        
        Args:
            transactions: Raw transactions data
            retailer_products: Product information
            prices: Pricing data
            
        Returns:
            Merged and cleaned DataFrame for analysis
        """
        try:
            # Merge datasets
            merged_df = pd.merge(
                transactions,
                retailer_products,
                on="product_id",
                how="left"
            )
            merged_df = pd.merge(
                merged_df,
                prices,
                on=["product_id", "date"],
                how="left"
            )
            
            # Convert date column to datetime
            merged_df["date"] = pd.to_datetime(merged_df["date"])
            
            # Handle missing values
            merged_df["quantity"] = merged_df["quantity"].fillna(0)
            merged_df["price"] = merged_df["price"].fillna(
                merged_df.groupby("product_id")["price"].transform("median")
            )
            
            logger.info("Data preprocessing completed successfully")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            raise
    
    def analyze_trends(self, df: pd.DataFrame) -> None:
        """
        Identify trending and declining products using short/long windows.
        Updates insights dictionary with results.
        
        Args:
            df: Preprocessed DataFrame for analysis
        """
        try:
            # Define window sizes (in days)
            short_window = 30
            long_window = 90
            
            # Get trending/declining products
            trending = DataFilters.identify_trending_products(
                df, 
                window_size=short_window,
                comparison_window=long_window
            )
            declining = DataFilters.identify_declining_products(
                df,
                window_size=short_window,
                comparison_window=long_window
            )
            
            # Format results
            self.insights["trending_products"] = trending.head(10).to_dict("records")
            self.insights["declining_products"] = declining.head(10).to_dict("records")
            
            logger.info(f"Identified {len(trending)} trending and {len(declining)} declining products")
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            raise
    
    def analyze_seasonality(self, df: pd.DataFrame) -> None:
        """
        Detect seasonal demand patterns.
        Updates insights dictionary with results.
        
        Args:
            df: Preprocessed DataFrame for analysis
        """
        try:
            seasonal_products = DataFilters.detect_seasonal_demand(df)
            
            # Format results with peak months
            results = []
            for product_id, seasonality in seasonal_products.items():
                peak_months = [m+1 for m in seasonality["peak_months"]]  # Convert 0-11 to 1-12
                results.append({
                    "product_id": product_id,
                    "seasonality_strength": seasonality["strength"],
                    "peak_months": peak_months,
                    "average_sales": seasonality["average_sales"]
                })
            
            self.insights["seasonal_products"] = sorted(
                results, 
                key=lambda x: x["seasonality_strength"], 
                reverse=True
            )[:20]
            
            logger.info(f"Identified {len(results)} seasonal products")
            
        except Exception as e:
            logger.error(f"Error in seasonality analysis: {e}")
            raise
    
    def analyze_sales_velocity(self, df: pd.DataFrame) -> None:
        """
        Calculate sales velocity and detect sudden changes.
        Updates insights dictionary with results.
        
        Args:
            df: Preprocessed DataFrame for analysis
        """
        try:
            velocity_changes = DataFilters.detect_sales_velocity_changes(df)
            
            # Format results
            self.insights["sales_velocity_changes"] = velocity_changes.to_dict("records")
            
            logger.info(f"Analyzed sales velocity for {len(velocity_changes)} products")
            
        except Exception as e:
            logger.error(f"Error in sales velocity analysis: {e}")
            raise
    
    def generate_insights(self) -> Dict[str, Any]:
        """
        Generate complete demand insights from available data.
        
        Returns:
            Dictionary containing all demand insights
        """
        try:
            # Load and prepare data
            transactions, retailer_products, prices = self.load_data()
            analysis_df = self.preprocess_data(transactions, retailer_products, prices)
            
            # Set metadata
            min_date = analysis_df["date"].min().strftime("%Y-%m-%d")
            max_date = analysis_df["date"].max().strftime("%Y-%m-%d")
            
            self.insights["metadata"] = {
                "analysis_date": datetime.now().strftime("%Y-%m-%d"),
                "data_range": f"{min_date} to {max_date}",
                "total_products": analysis_df["product_id"].nunique(),
                "total_transactions": len(analysis_df)
            }
            
            # Perform analyses
            self.analyze_trends(analysis_df)
            self.analyze_seasonality(analysis_df)
            self.analyze_sales_velocity(analysis_df)
            
            logger.info("Successfully generated all demand insights")
            return self.insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            raise
    
    def save_insights(self, output_path: str = "data/processed/demand_insights.json") -> None:
        """
        Save generated insights to JSON file.
        
        Args:
            output_path: Path to save JSON output
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, "w") as f:
                json.dump(self.insights, f, indent=2)
            
            logger.info(f"Saved demand insights to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving insights: {e}")
            raise

if __name__ == "__main__":
    try:
        logger.info("Starting demand analysis...")
        
        analyzer = DemandAnalyzer()
        insights = analyzer.generate_insights()
        analyzer.save_insights()
        
        logger.info("Demand analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Demand analysis failed: {e}")
        raise