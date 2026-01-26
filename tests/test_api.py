"""
Unit tests for the pricing API
"""

import unittest
import json
from fastapi.testclient import TestClient
from src.api import app, PricingRequest

class TestAPI(unittest.TestCase):
    
    def setUp(self):
        self.client = TestClient(app)
        
        # Train model for testing
        from src.model import train_pricing_model
        train_pricing_model(
            data_path='data/simulated_data.csv',
            model_path='models/test_lgbm_model.pkl',
            product_combinations_path='models/test_product_combinations.pkl'
        )
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
        self.assertIn("GPBS Pricing Simulator API", response.json()["message"])
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")
    
    def test_prediction(self):
        """Test prediction endpoint"""
        request_data = {
            "country": "SG",
            "segment_code": 600,
            "tier": "Gold",
            "volume": 5000,
            "currency": "USD"
        }
        
        response = self.client.post("/predict", json=request_data)
        self.assertEqual(response.status_code, 200)
        
        # Check response structure
        response_data = response.json()
        self.assertIn("pricing", response_data)
        self.assertIn("metadata", response_data)
        
        pricing = response_data["pricing"]
        self.assertIsInstance(pricing, dict)
        
        # Should have at least one product level 1
        self.assertGreater(len(pricing), 0)
        
        # Check product code differentiation within same product level 2
        for category, items in pricing.items():
            for item in items:
                for subcategory, products in item.items():
                    if len(products) > 1:
                        prices = [float(p) for p in products.values()]
                        self.assertGreater(max(prices) - min(prices), 0.0001, 
                                          f"Prices for {subcategory} should differ: {products}")
    
    def test_tier_discounts(self):
        """Test tier-based pricing differences"""
        # Get Gold tier pricing
        gold_request = {
            "country": "SG",
            "segment_code": 600,
            "tier": "Gold",
            "volume": 5000,
            "currency": "USD"
        }
        gold_response = self.client.post("/predict", json=gold_request)
        gold_pricing = gold_response.json()["pricing"]
        
        # Get Silver tier pricing
        silver_request = {
            "country": "SG",
            "segment_code": 600,
            "tier": "Silver",
            "volume": 5000,
            "currency": "USD"
        }
        silver_response = self.client.post("/predict", json=silver_request)
        silver_pricing = silver_response.json()["pricing"]
        
        # Compare prices - Gold should be cheaper than Silver
        gold_avg = self.get_average_price(gold_pricing)
        silver_avg = self.get_average_price(silver_pricing)
        
        self.assertLess(gold_avg, silver_avg, 
                       f"Gold pricing ({gold_avg}) should be less than Silver pricing ({silver_avg})")
        self.assertGreaterEqual((silver_avg - gold_avg) / silver_avg, 0.05,
                               "Gold discount should be at least 5%")
    
    def get_average_price(self, pricing):
        """Helper to calculate average price from pricing structure"""
        total = 0
        count = 0
        for category, items in pricing.items():
            for item in items:
                for subcategory, products in item.items():
                    for price_str in products.values():
                        total += float(price_str)
                        count += 1
        return total / count if count > 0 else 0
    
    def test_singapore_pricing(self):
        """Test Singapore-specific pricing features"""
        # Get SG pricing
        sg_request = {
            "country": "SG",
            "segment_code": 600,
            "tier": "Gold",
            "volume": 5000,
            "currency": "USD"
        }
        sg_response = self.client.post("/predict", json=sg_request)
        sg_pricing = sg_response.json()["pricing"]
        
        # Get US pricing
        us_request = {
            "country": "US",
            "segment_code": 600,
            "tier": "Gold",
            "volume": 5000,
            "currency": "USD"
        }
        us_response = self.client.post("/predict", json=us_request)
        us_pricing = us_response.json()["pricing"]
        
        # Compare prices - SG should be cheaper than US for Payments and Clearing
        sg_avg = self.get_average_price(sg_pricing)
        us_avg = self.get_average_price(us_pricing)
        
        # Payments and Clearing should be cheaper in SG
        sg_payments = self.get_category_average(sg_pricing, "Payments")
        us_payments = self.get_category_average(us_pricing, "Payments")
        self.assertLess(sg_payments, us_payments, 
                       "Payments in SG should be cheaper than in US")
        
        sg_clearing = self.get_category_average(sg_pricing, "Clearing")
        us_clearing = self.get_category_average(us_pricing, "Clearing")
        self.assertLess(sg_clearing, us_clearing, 
                       "Clearing in SG should be cheaper than in US")
    
    def get_category_average(self, pricing, category):
        """Helper to calculate average price for a specific category"""
        if category not in pricing:
            return 0
            
        total = 0
        count = 0
        for item in pricing[category]:
            for subcategory, products in item.items():
                for price_str in products.values():
                    total += float(price_str)
                    count += 1
        return total / count if count > 0 else 0
    
    def test_price_structure_validation(self):
        """Test that price structure follows business rules"""
        request_data = {
            "country": "SG",
            "segment_code": 600,
            "tier": "Gold",
            "volume": 5000,
            "currency": "USD"
        }
        
        response = self.client.post("/predict", json=request_data)
        pricing = response.json()["pricing"]
        
        # Rule 1: Clearing > Payments > Collections > Liquidity
        clearing_avg = self.get_category_average(pricing, "Clearing")
        payments_avg = self.get_category_average(pricing, "Payments")
        collections_avg = self.get_category_average(pricing, "Collections")
        liquidity_avg = self.get_category_average(pricing, "Liquidity")
        
        self.assertGreaterEqual(clearing_avg, payments_avg)
        self.assertGreaterEqual(payments_avg, collections_avg)
        self.assertGreaterEqual(collections_avg, liquidity_avg)
    
    def test_explain_endpoint(self):
        """Test feature explanation endpoint"""
        # Test valid feature
        response = self.client.get("/explain?feature=service_level_factor")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["feature"], "service_level_factor")
        self.assertIn("STD=1.0", data["description"])
        
        # Test invalid feature
        response = self.client.get("/explain?feature=invalid_feature")
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
