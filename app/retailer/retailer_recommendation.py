#!/usr/bin/env python3
"""
Retailer recommendation engine for market_trend_app.
Generates data-driven recommendations for inventory and pricing.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
from utils.csv_loader import CSVLoader
from utils.filters import DataFilters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RetailerRecommendation:
    """Generates data-driven recommendations for retailers."""
    
    def __init__(self, data_dir: str = "data/raw", processed_dir: str = "data/processed"):
        """
        Initialize with data directories.
        
        Args:
            data_dir: Directory containing raw CSV files
            processed_dir: Directory containing processed data (JSON)
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.recommendations = {
            "metadata": {"analysis_date": datetime.now().strftime("%Y-%m-%d")},
            "restock_suggestions": [],
            "discount_suggestions": [],
            "price_optimizations": [],
            "seasonal_stocking": []
        }
        self.demand_insights: Dict[str, Any] = {}
        self.retailer_products: pd.DataFrame = pd.DataFrame()
        self.prices: pd.DataFrame = pd.DataFrame()
        
    def load_data(self) -> None:
        """Load all required data files."""
        try:
            # Load processed JSON data
            demand_file = self.processed_dir / "demand_insights.json"
            with open(demand_file, 'r') as f:
                self.demand_insights = json.load(f)
            
            # Load CSV data
            loader = CSVLoader(self.data_dir)
            self.retailer_products = loader.load_csv("retailer_products.csv")
            self.prices = loader.load_csv("prices.csv")
            
            logger.info("Successfully loaded all data files")
            
        except FileNotFoundError as e:
            logger.error(f"Missing required data file: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in demand_insights.json: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _preprocess_data(self) -> None:
        """Clean and prepare data for analysis."""
        try:
            # Clean retailer products
            self.retailer_products = self.retailer_products.dropna(subset=['product_id'])
            self.retailer_products['current_stock'] = pd.to_numeric(
                self.retailer_products['current_stock'], errors='coerce'
            ).fillna(0)
            self.retailer_products['current_price'] = pd.to_numeric(
                self.retailer_products['current_price'], errors='coerce'
            ).fillna(0)
            
            # Clean prices data
            self.prices = self.prices.dropna(subset=['product_id', 'price'])
            self.prices['price'] = pd.to_numeric(self.prices['price'], errors='coerce')
            self.prices = self.prices[self.prices['price'] > 0]
            
            logger.info("Data preprocessing completed")
            
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            raise
    
    def _recommend_restock(self) -> None:
        """Identify products needing restock based on demand and current stock."""
        try:
            if not self.demand_insights.get("trending_products"):
                logger.warning("No trending products data available")
                return
            
            # Get trending products
            trending_df = pd.DataFrame(self.demand_insights["trending_products"])
            trending_products = trending_df["product_id"].unique()
            
            # Merge with retailer inventory
            inventory_df = self.retailer_products[
                self.retailer_products["product_id"].isin(trending_products)
            ].copy()
            
            if inventory_df.empty:
                logger.warning("No matching trending products in retailer inventory")
                return
            
            # Calculate median stock levels for similar products
            median_stock = self.retailer_products.groupby("category")["current_stock"].median().reset_index()
            median_stock.columns = ["category", "median_category_stock"]
            
            # Merge and calculate recommendations
            inventory_df = pd.merge(inventory_df, median_stock, on="category", how="left")
            inventory_df["recommended_stock"] = inventory_df["median_category_stock"].apply(
                lambda x: max(x, 20)  # Minimum recommended stock of 20
            )
            
            # Filter for low stock items
            restock_df = inventory_df[
                inventory_df["current_stock"] < (0.3 * inventory_df["recommended_stock"])
            ].sort_values("current_stock")
            
            # Format results
            self.recommendations["restock_suggestions"] = restock_df[
                ["product_id", "product_name", "current_stock", "recommended_stock"]
            ].rename(columns={
                "product_id": "product_id",
                "product_name": "product",
                "current_stock": "current_stock",
                "recommended_stock": "recommended_stock"
            }).to_dict("records")
            
            logger.info(f"Generated {len(restock_df)} restock suggestions")
            
        except Exception as e:
            logger.error(f"Error in restock recommendations: {e}")
            raise
    
    def _recommend_discounts(self) -> None:
        """Identify products that should be discounted."""
        try:
            # Get declining products and high stock items
            declining_products = pd.DataFrame(
                self.demand_insights.get("declining_products", [])
            )
            high_stock_items = self.retailer_products[
                self.retailer_products["current_stock"] > 
                self.retailer_products.groupby("category")["current_stock"].transform("median") * 2
            ]
            
            # Combine discount candidates
            discount_candidates = pd.concat([
                declining_products[["product_id"]],
                high_stock_items[["product_id", "product_name", "current_stock"]]
            ]).drop_duplicates()
            
            if discount_candidates.empty:
                logger.warning("No discount candidates found")
                return
            
            # Merge with product info
            discount_df = pd.merge(
                discount_candidates,
                self.retailer_products,
                on="product_id",
                how="left"
            )
            
            # Format results with reasons
            discount_suggestions = []
            for _, row in discount_df.iterrows():
                suggestion = {
                    "product_id": row["product_id"],
                    "product": row["product_name"],
                    "current_stock": row["current_stock"],
                    "reason": ""
                }
                
                if row["product_id"] in declining_products["product_id"].values:
                    suggestion["reason"] = "Declining demand"
                else:
                    suggestion["reason"] = "High stock surplus"
                
                discount_suggestions.append(suggestion)
            
            self.recommendations["discount_suggestions"] = discount_suggestions
            logger.info(f"Generated {len(discount_suggestions)} discount suggestions")
            
        except Exception as e:
            logger.error(f"Error in discount recommendations: {e}")
            raise
    
    def _recommend_price_adjustments(self) -> None:
        """Suggest price optimizations based on market prices."""
        try:
            # Calculate average prices
            avg_prices = self.prices.groupby("product_id")["price"].mean().reset_index()
            avg_prices.columns = ["product_id", "market_avg_price"]
            
            # Merge with retailer prices
            price_df = pd.merge(
                self.retailer_products,
                avg_prices,
                on="product_id",
                how="inner"
            )
            
            if price_df.empty:
                logger.warning("No products with matching price data")
                return
            
            # Calculate price differences and recommendations
            price_df["price_diff_pct"] = (
                (price_df["current_price"] - price_df["market_avg_price"]) / 
                price_df["market_avg_price"] * 100
            )
            
            # Apply pricing rules
            def calculate_suggested_price(row):
                if row["price_diff_pct"] > 15:  # Overpriced
                    return row["market_avg_price"] * 1.05  # 5% above market avg
                elif row["price_diff_pct"] < -10:  # Underpriced
                    return min(
                        row["market_avg_price"] * 0.95,  # 5% below market avg
                        row["current_price"] * 1.10  # But max 10% increase
                    )
                return row["current_price"]  # No change
            
            price_df["suggested_price"] = price_df.apply(calculate_suggested_price, axis=1)
            
            # Filter for products needing adjustment
            price_adj_df = price_df[
                abs(price_df["suggested_price"] - price_df["current_price"]) > 0.01
            ]
            
            # Format results
            self.recommendations["price_optimizations"] = price_adj_df[
                ["product_id", "product_name", "current_price", "market_avg_price", "suggested_price"]
            ].rename(columns={
                "product_id": "product_id",
                "product_name": "product",
                "current_price": "current_price",
                "market_avg_price": "average_price",
                "suggested_price": "suggested_price"
            }).to_dict("records")
            
            logger.info(f"Generated {len(price_adj_df)} price optimization suggestions")
            
        except Exception as e:
            logger.error(f"Error in price recommendations: {e}")
            raise
    
    def _recommend_seasonal_stocking(self) -> None:
        """Suggest seasonal stock adjustments."""
        try:
            seasonal_products = self.demand_insights.get("seasonal_products", [])
            if not seasonal_products:
                logger.warning("No seasonal products data available")
                return
            
            # Get current month
            current_month = datetime.now().month
            
            # Find seasonal products approaching peak
            seasonal_recs = []
            for product in seasonal_products:
                peak_months = product.get("peak_months", [])
                months_until_peak = min(
                    (m - current_month) % 12 for m in peak_months
                )
                
                # Recommend if peak is within 2 months
                if months_until_peak <= 2:
                    # Calculate suggested stock based on average sales
                    suggested_stock = product["average_sales"] * 1.5  # 50% buffer
                    seasonal_recs.append({
                        "product_id": product["product_id"],
                        "product": self._get_product_name(product["product_id"]),
                        "peak_month": datetime(2000, peak_months[0], 1).strftime("%B"),
                        "stock_suggestion": round(suggested_stock)
                    })
            
            self.recommendations["seasonal_stocking"] = seasonal_recs
            logger.info(f"Generated {len(seasonal_recs)} seasonal stocking suggestions")
            
        except Exception as e:
            logger.error(f"Error in seasonal recommendations: {e}")
            raise
    
    def _get_product_name(self, product_id: str) -> Optional[str]:
        """Helper to get product name from ID."""
        try:
            product = self.retailer_products[
                self.retailer_products["product_id"] == product_id
            ].iloc[0]
            return product["product_name"]
        except (IndexError, KeyError):
            logger.warning(f"Could not find name for product {product_id}")
            return None
    
    def generate_recommendations(self) -> Dict[str, Any]:
        """
        Generate all retailer recommendations.
        
        Returns:
            Dictionary containing all recommendations
        """
        try:
            self.load_data()
            self._preprocess_data()
            
            self._recommend_restock()
            self._recommend_discounts()
            self._recommend_price_adjustments()
            self._recommend_seasonal_stocking()
            
            logger.info("Successfully generated all recommendations")
            return self.recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            raise
    
    def save_recommendations(self, output_path: str = "data/processed/retailer_recommendations.json") -> None:
        """
        Save recommendations to JSON file.
        
        Args:
            output_path: Path to save JSON output
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(self.recommendations, f, indent=2)
            
            logger.info(f"Saved recommendations to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving recommendations: {e}")
            raise

if __name__ == "__main__":
    try:
        logger.info("Starting retailer recommendation engine...")
        
        recommender = RetailerRecommendation()
        recommendations = recommender.generate_recommendations()
        recommender.save_recommendations()
        
        logger.info("Retailer recommendations completed successfully")
        
    except Exception as e:
        logger.error(f"Retailer recommendation process failed: {e}")
        raise