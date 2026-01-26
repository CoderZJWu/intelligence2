"""
Performs feature engineering for the GPBS pricing model.
Prepares data for LightGBM training by handling categorical features
and creating relevant pricing features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_and_prepare_data(file_path='data/simulated_data.csv'):
    """Load data and prepare features for model training"""
    df = pd.read_csv(file_path)
    
    # Create volume per transaction feature
    df['volume_per_transaction'] = df['FY24_volume'] / (df['FY24_fee_in_usd'] / df['ecp_usd'])
    
    # Create revenue growth features
    df['revenue_growth'] = (df['FY24_fee_in_usd'] - df['FY23_fee_in_usd']) / df['FY23_fee_in_usd']
    df['volume_growth'] = (df['FY24_volume'] - df['FY23_volume']) / df['FY23_volume']
    
    # Create service level features (NEW)
    df['is_premium_service'] = df['service_level'].apply(lambda x: 1 if x in ['PRM', 'VIP', 'ELT'] else 0)
    df['service_level_factor'] = df['service_level'].map({
        'STD': 1.0,
        'PRM': 1.2,
        'VIP': 1.45,
        'ELT': 1.75
    })
    
    # Create country-segment interaction feature (NEW)
    df['country_segment'] = df['country'] + '_' + df['segment_name'].str.replace(' ', '_')
    
    # Handle infinite values
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def create_feature_sets(df):
    """Create feature sets for model training"""
    # Target variable
    y = df['ecp_usd']
    
    # Features to use
    features = [
        'country', 'segment_code', 'tier', 'product_level1', 'product_level2', 
        'charge_currency', 'computation_method', 'FY24_volume', 'volume_per_transaction',
        'revenue_growth', 'volume_growth', 'avg_cp_at_country', 'avg_cp_at_seg',
        'avg_cp_at_tier', 'avg_cp_at_wa_volume', 'service_level_factor',  # NEW
        'is_premium_service', 'country_segment'  # NEW
    ]
    
    X = df[features].copy()
    
    # Encode categorical features
    cat_features = ['country', 'segment_code', 'tier', 'product_level1', 
                    'product_level2', 'charge_currency', 'computation_method',
                    'country_segment']  # NEW
    
    # Convert segment_code to categorical (even though it's numeric)
    X['segment_code'] = X['segment_code'].astype('category')
    
    # Label encode categorical features
    label_encoders = {}
    for feature in cat_features:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature])
        label_encoders[feature] = le
    
    return X, y, label_encoders, cat_features

if __name__ == "__main__":
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    print("Creating feature sets...")
    X, y, label_encoders, cat_features = create_feature_sets(df)
    
    print("\nFeature preparation complete!")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    print("\nCategorical features encoded:", cat_features)
    print("\nFirst 5 rows of processed features:")
    print(X.head())
