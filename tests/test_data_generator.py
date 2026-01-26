"""
Unit tests for data_generator.py
"""

import unittest
import os
import pandas as pd
from src.data_generator import generate_simulated_data

class TestDataGenerator(unittest.TestCase):
    
    def setUp(self):
        self.test_file = 'data/test_simulated_data.csv'
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_data_generation(self):
        """Test if data is generated correctly"""
        df = generate_simulated_data(n_samples=100)
        
        # Check file exists
        self.assertTrue(os.path.exists('data/simulated_data.csv'))
        
        # Check DataFrame shape
        self.assertEqual(df.shape[0], 100)
        self.assertGreater(df.shape[1], 10)  # Should have more than 10 columns
        
        # Check required columns exist
        required_columns = [
            'country', 'segment_code', 'tier', 'product_level1',
            'ecp_usd', 'FY24_volume', 'charge_currency'
        ]
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Check ecp_usd values are reasonable
        self.assertTrue((df['ecp_usd'] > 0).all())
        self.assertTrue((df['ecp_usd'] < 5.0).all())
    
    def tearDown(self):
        if os.path.exists('data/simulated_data.csv'):
            os.remove('data/simulated_data.csv')

if __name__ == '__main__':
    unittest.main()
