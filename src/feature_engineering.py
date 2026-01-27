"""
Performs feature engineering for the GPBS pricing model.
Prepares data for LightGBM training by handling categorical features
and creating relevant pricing features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re

def extract_product_code_patterns(df):
    """
    自动从product_code中提取模式特征，无需业务知识
    """
    # 1. 提取产品代码长度特征（可能反映复杂度）
    df['product_code_length'] = df['product_code'].str.len()
    
    # 2. 提取数字部分作为版本特征
    df['product_version'] = df['product_code'].str.extract('([0-9]+)', expand=False).astype(float)
    
    # 3. 提取字母模式特征（自动识别重复模式）
    df['product_pattern'] = df['product_code'].str.replace(r'[0-9]', '', regex=True)
    
    # 4. 计算字母变化率（可能反映服务级别）
    def calculate_letter_change_rate(code):
        changes = 0
        for i in range(1, len(code)):
            if code[i] != code[i-1]:
                changes += 1
        return changes / len(code) if len(code) > 0 else 0
    
    df['letter_change_rate'] = df['product_code'].apply(calculate_letter_change_rate)
    
    # 5. 提取产品代码的字符分布特征
    df['has_upper'] = df['product_code'].str.isupper().astype(int)
    df['has_digits'] = df['product_code'].str.contains(r'[0-9]').astype(int)
    df['has_special'] = df['product_code'].str.contains(r'[^A-Za-z0-9]').astype(int)
    
    # 6. 创建产品代码的嵌入特征（使用简单哈希）
    df['product_code_hash'] = df['product_code'].apply(lambda x: hash(x) % 1000) / 1000.0
    
    return df

def load_and_prepare_data(file_path='data/simulated_data.csv'):
    """Load data and prepare features for model training"""
    df = pd.read_csv(file_path)
    
    # Create volume per transaction feature
    df['volume_per_transaction'] = df['FY24_volume'] / (df['FY24_fee_in_usd'] / df['ecp_usd'])
    
    # Create revenue growth features
    df['revenue_growth'] = (df['FY24_fee_in_usd'] - df['FY23_fee_in_usd']) / df['FY23_fee_in_usd']
    df['volume_growth'] = (df['FY24_volume'] - df['FY23_volume']) / df['FY23_volume']
    
    df['country_segment'] = df['country'] + '_' + df['segment_name'].str.replace(' ', '_')
    # Handle infinite values
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    
    # Extract patterns from product code
    df = extract_product_code_patterns(df)
    
    return df

def create_feature_sets(df):
    """Create feature sets for model training"""
    # Target variable
    y = df['ecp_usd']
    
    # Features to use (removed service_level_factor and is_premium_service)
    features = [
        'country', 'segment_code', 'tier', 'product_level1', 'product_level2', 
        'charge_currency', 'computation_method', 'FY24_volume', 'volume_per_transaction',
        'revenue_growth', 'volume_growth', 'avg_cp_at_country', 'avg_cp_at_seg',
        'avg_cp_at_tier', 'avg_cp_at_wa_volume', 'country_segment',
        # 新增的自动提取特征
        'product_code_length', 'product_version', 'letter_change_rate',
        'has_upper', 'has_digits', 'has_special', 'product_code_hash'
    ]
    
    X = df[features].copy()
    
    # Encode categorical features
    cat_features = ['country', 'segment_code', 'tier', 'product_level1', 
                    'product_level2', 'charge_currency', 'computation_method',
                    'country_segment', 'product_pattern']  # 新增product_pattern
    
    # Convert segment_code to categorical (even though it's numeric)
    X['segment_code'] = X['segment_code'].astype('category')
    
    # Label encode categorical features
    label_encoders = {}
    for feature in cat_features:
        if feature in X.columns:
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
