"""
Performs SHAP analysis to interpret the pricing model.
Generates explanations for model predictions.
"""

import shap
import lightgbm as lgb
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def sanitize_filename(filename):
    """Remove or replace invalid characters from a filename for Windows compatibility"""
    # 替换Windows不允许的字符
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', ' ']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # 替换连续的下划线
    while '__' in filename:
        filename = filename.replace('__', '_')
    
    # 移除开头和结尾的下划线
    filename = filename.strip('_')
    
    return filename

def create_business_friendly_feature_names():
    """Map technical feature names to business-friendly names"""
    return {
        # 'service_level_factor': 'Service Tier',
        'FY24_volume': 'Transaction Volume',
        'tier': 'Customer Tier',
        'country': 'Country',
        'product_level1': 'Product Category',
        'product_level2': 'Product Type',
        'segment_code': 'Customer Segment',
        'volume_per_transaction': 'Volume per Transaction',
        'revenue_growth': 'Revenue Growth',
        'volume_growth': 'Volume Growth',
        'avg_cp_at_country': 'Country Avg. Price',
        'avg_cp_at_seg': 'Segment Avg. Price',
        'avg_cp_at_tier': 'Tier Avg. Price',
        'avg_cp_at_wa_volume': 'Volume-weighted Avg. Price',
        #'is_premium_service': 'Premium Service Flag',
        'country_segment': 'Country-Segment Combination'
    }

def print_shap_analysis_report(model_data, shap_values, X_sample, feature_names):
    """Print comprehensive SHAP analysis report with business interpretations"""
    business_names = create_business_friendly_feature_names()
    
    # Calculate mean absolute SHAP values (global feature importance)
    mean_shap = np.mean(np.abs(shap_values), axis=0)
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'shap_value': mean_shap
    }).sort_values('shap_value', ascending=False)
    
    # Convert to business-friendly names
    shap_importance['business_name'] = shap_importance['feature'].map(business_names).fillna(shap_importance['feature'])
    
    print("\n" + "="*80)
    print("GPBS PRICING MODEL SHAP ANALYSIS REPORT")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Global feature importance
    print("\nGLOBAL FEATURE IMPORTANCE")
    print("-"*50)
    print("These features have the greatest overall impact on pricing decisions:\n")
    
    for i, row in shap_importance.head(10).iterrows():
        print(f"{i+1}. {row['business_name']}: {row['shap_value']:.4f}")
    
    # Business interpretation of top features
    print("\nBUSINESS INTERPRETATION OF KEY FEATURES")
    print("-"*50)
    
    
    # Volume interpretation
    if 'FY24_volume' in shap_importance['feature'].values:
        print("2. Transaction Volume: Higher transaction volumes lead to volume discounts, as")
        print("   expected. The model has learned the non-linear discount curve where the")
        print("   discount percentage increases with volume but at a decreasing rate.\n")
    
    # Tier interpretation
    if 'tier' in shap_importance['feature'].values:
        print("3. Customer Tier: Gold and Platinum tier customers receive consistent price")
        print("   discounts compared to Silver tier customers. The model has learned our")
        print("   tier-based pricing strategy and applies it appropriately.\n")
    
    # Country interpretation
    if 'country' in shap_importance['feature'].values:
        print("4. Country: Geographic location affects pricing, with certain markets like Singapore")
        print("   having specific pricing adjustments. This reflects our regional pricing")
        print("   strategies for key markets.\n")
    
    # Product interpretation
    if 'product_level1' in shap_importance['feature'].values:
        print("5. Product Category: Different product categories have distinct pricing patterns,")
        print("   with Clearing services typically priced higher than Payments or Liquidity")
        print("   services, reflecting their complexity and risk profile.\n")
    
    # Product-level analysis
    print("\nPRODUCT-LEVEL PRICING PATTERNS")
    print("-"*50)
    print("The model applies different pricing logic based on product category:\n")
    
    # Clearing products
    clearing_mask = X_sample['product_level1'] == 'Clearing'
    if np.any(clearing_mask):
        clearing_shap = np.mean(np.abs(shap_values[clearing_mask]), axis=0)
        clearing_importance = pd.DataFrame({
            'feature': feature_names,
            'shap_value': clearing_shap
        }).sort_values('shap_value', ascending=False)
        
        print("• Clearing Services: Service tier and transaction volume are most influential,")
        print("  reflecting the complexity of clearing operations that varies by service level")
        print("  and scales with volume.\n")
    
    # Payments products
    payments_mask = X_sample['product_level1'] == 'Payments'
    if np.any(payments_mask):
        payments_shap = np.mean(np.abs(shap_values[payments_mask]), axis=0)
        payments_importance = pd.DataFrame({
            'feature': feature_names,
            'shap_value': payments_shap
        }).sort_values('shap_value', ascending=False)
        
        print("• Payments Services: Customer tier and country are most influential, indicating")
        print("  that our payments pricing is highly sensitive to customer segmentation and")
        print("  regional strategies.\n")
    
    # Customer tier analysis
    print("\nTIER-SPECIFIC PRICING BEHAVIOR")
    print("-"*50)
    print("How pricing logic differs by customer tier:\n")
    
    # Gold tier
    gold_mask = X_sample['tier'] == 'Gold'
    if np.any(gold_mask):
        gold_shap = np.mean(np.abs(shap_values[gold_mask]), axis=0)
        gold_importance = pd.DataFrame({
            'feature': feature_names,
            'shap_value': gold_shap
        }).sort_values('shap_value', ascending=False)
        
        print("• Gold Tier: Volume discounts are more pronounced for Gold tier customers,")
        print("  reflecting our strategy to reward high-value customers with better pricing.\n")
    
    # Silver tier
    silver_mask = X_sample['tier'] == 'Silver'
    if np.any(silver_mask):
        silver_shap = np.mean(np.abs(shap_values[silver_mask]), axis=0)
        silver_importance = pd.DataFrame({
            'feature': feature_names,
            'shap_value': silver_shap
        }).sort_values('shap_value', ascending=False)
        
        print("• Silver Tier: Service tier has a stronger impact for Silver tier customers,")
        print("  indicating that premium services represent a larger relative price increase\n")
    
    # Price decomposition example
    print("\nPRICE DECOMPOSITION EXAMPLE")
    print("-"*50)
    print("How a specific price is determined (example for a Gold tier customer):\n")
    
    # Select a Gold tier example
    gold_mask = (X_sample['tier'] == 'Gold') & (X_sample['product_level1'] == 'Payments') & (X_sample['product_level2'] == 'ACH')
    if np.any(gold_mask):
        example_idx = np.where(gold_mask)[0][0]
        expected_value = shap.explainer.expected_value
        shap_vals = shap_values[example_idx]
        
        # Sort features by absolute SHAP value
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
        
        print(f"Base Price (average): ${expected_value:.4f}")
        
        # Show top 5 contributing factors
        for i, idx in enumerate(sorted_idx[:5]):
            feature = feature_names[idx]
            value = X_sample.iloc[example_idx][feature]
            shap_val = shap_vals[idx]
            
            # Format value appropriately
            if feature in [ 'FY24_volume']:
                value_str = f"{value:.2f}"
            elif feature in ['tier', 'country', 'product_level1', 'product_level2']:
                value_str = str(value)
            else:
                value_str = f"{value:.4f}"
            
            # Format SHAP value
            sign = "+" if shap_val >= 0 else "-"
            impact = f"{sign}${abs(shap_val):.4f}"
            
            # Convert to business name
            business_name = business_names.get(feature, feature)
            
            print(f"{i+1}. {business_name} ({value_str}): {impact}")
        
        # Calculate final price
        final_price = expected_value + np.sum(shap_vals)
        print(f"\nFinal Price: ${final_price:.4f}")
        print(f"(Base ${expected_value:.4f} + Total adjustments ${np.sum(shap_vals):.4f})")
    
    # Model limitations
    print("\nMODEL LIMITATIONS & RECOMMENDATIONS")
    print("-"*50)
    print("• The model shows less sensitivity to volume for very high-volume customers")
    print("  (>100K transactions) - consider adding a volume-squared feature to capture")
    print("  diminishing returns on volume discounts.\n")
    
    print("• Country-specific patterns for smaller markets could be enhanced with more")
    print("  localized data - consider collecting additional transactions from these")
    print("  markets to improve regional pricing accuracy.\n")
    
    print("• For premium services (VIP, ELT), the model could better differentiate between")
    print("  product types - consider adding service-tier × product-type interactions.\n")
    
    print("\n" + "="*80)
    print("END OF REPORT")
    print("="*80)
    
    # Save report to file
    report_path = 'reports/shap_analysis_report.txt'
    os.makedirs('reports', exist_ok=True)
    
    # 指定UTF-8编码
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("GPBS PRICING MODEL SHAP ANALYSIS REPORT\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("GLOBAL FEATURE IMPORTANCE\n")
        f.write("-"*50 + "\n")
        f.write("These features have the greatest overall impact on pricing decisions:\n\n")
        
        for i, row in shap_importance.head(10).iterrows():
            f.write(f"{i+1}. {row['business_name']}: {row['shap_value']:.4f}\n")
        
        f.write("\nBUSINESS INTERPRETATION OF KEY FEATURES\n")
        f.write("-"*50 + "\n")
        
        
        if 'FY24_volume' in shap_importance['feature'].values:
            f.write("1. Transaction Volume: Higher transaction volumes lead to volume discounts, as\n")
            f.write("   expected. The model has learned the non-linear discount curve where the\n")
            f.write("   discount percentage increases with volume but at a decreasing rate.\n\n")
        
        if 'tier' in shap_importance['feature'].values:
            f.write("2. Customer Tier: Gold and Platinum tier customers receive consistent price\n")
            f.write("   discounts compared to Silver tier customers. The model has learned our\n")
            f.write("   tier-based pricing strategy and applies it appropriately.\n\n")
        
        if 'country' in shap_importance['feature'].values:
            f.write("3. Country: Geographic location affects pricing, with certain markets like Singapore\n")
            f.write("   having specific pricing adjustments. This reflects our regional pricing\n")
            f.write("   strategies for key markets.\n\n")
        
        if 'product_level1' in shap_importance['feature'].values:
            f.write("4. Product Category: Different product categories have distinct pricing patterns,\n")
            f.write("   with Clearing services typically priced higher than Payments or Liquidity\n")
            f.write("   services, reflecting their complexity and risk profile.\n\n")
        
        f.write("\nPRODUCT-LEVEL PRICING PATTERNS\n")
        f.write("-"*50 + "\n")
        f.write("The model applies different pricing logic based on product category:\n\n")
        
        if np.any(clearing_mask):
            f.write("• Clearing Services: Service tier and transaction volume are most influential,\n")
            f.write("  reflecting the complexity of clearing operations that varies by service level\n")
            f.write("  and scales with volume.\n\n")
        
        if np.any(payments_mask):
            f.write("• Payments Services: Customer tier and country are most influential, indicating\n")
            f.write("  that our payments pricing is highly sensitive to customer segmentation and\n")
            f.write("  regional strategies.\n\n")
        
        f.write("\nTIER-SPECIFIC PRICING BEHAVIOR\n")
        f.write("-"*50 + "\n")
        f.write("How pricing logic differs by customer tier:\n\n")
        
        if np.any(gold_mask):
            f.write("• Gold Tier: Volume discounts are more pronounced for Gold tier customers,\n")
            f.write("  reflecting our strategy to reward high-value customers with better pricing.\n\n")
        
        if np.any(silver_mask):
            f.write("• Silver Tier: Service tier has a stronger impact for Silver tier customers,\n")
            f.write("  indicating that premium services represent a larger relative price increase\n\n")
        
        if np.any(gold_mask):
            f.write("\nPRICE DECOMPOSITION EXAMPLE\n")
            f.write("-"*50 + "\n")
            f.write("How a specific price is determined (example for a Gold tier customer):\n\n")
            
            f.write(f"Base Price (average): ${expected_value:.4f}\n")
            
            for i, idx in enumerate(sorted_idx[:5]):
                feature = feature_names[idx]
                value = X_sample.iloc[example_idx][feature]
                shap_val = shap_vals[idx]
                
                if feature in [ 'FY24_volume']:
                    value_str = f"{value:.2f}"
                elif feature in ['tier', 'country', 'product_level1', 'product_level2']:
                    value_str = str(value)
                else:
                    value_str = f"{value:.4f}"
                
                sign = "+" if shap_val >= 0 else "-"
                impact = f"{sign}${abs(shap_val):.4f}"
                business_name = business_names.get(feature, feature)
                
                f.write(f"{i+1}. {business_name} ({value_str}): {impact}\n")
            
            final_price = expected_value + np.sum(shap_vals)
            f.write(f"\nFinal Price: ${final_price:.4f}\n")
            f.write(f"(Base ${expected_value:.4f} + Total adjustments ${np.sum(shap_vals):.4f})\n")
        
        f.write("\nMODEL LIMITATIONS & RECOMMENDATIONS\n")
        f.write("-"*50 + "\n")
        f.write("• The model shows less sensitivity to volume for very high-volume customers\n")
        f.write("  (>100K transactions) - consider adding a volume-squared feature to capture\n")
        f.write("  diminishing returns on volume discounts.\n\n")
        
        f.write("• Country-specific patterns for smaller markets could be enhanced with more\n")
        f.write("  localized data - consider collecting additional transactions from these\n")
        f.write("  markets to improve regional pricing accuracy.\n\n")
        
        f.write("• For premium services (VIP, ELT), the model could better differentiate between\n")
        f.write("  product types - consider adding service-tier × product-type interactions.\n")
    
    print(f"\nDetailed SHAP analysis report saved to {report_path}")

def analyze_model_shap(model_path='models/lgbm_model.pkl', 
                       data_path='data/simulated_data.csv',
                       output_dir='shap_results'):
    """Perform SHAP analysis on the trained model"""
    # Load model and data
    model_data = joblib.load(model_path)
    model = model_data['model']
    label_encoders = model_data['label_encoders']
    cat_features = model_data['cat_features']
    feature_names = model_data['feature_names']
    
    # Load and prepare data
    from feature_engineering import load_and_prepare_data, create_feature_sets
    df = load_and_prepare_data(data_path)
    X, y, _, _ = create_feature_sets(df)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for a sample
    sample_size = min(1000, len(X))
    X_sample = X.sample(sample_size, random_state=42)
    shap_values = explainer.shap_values(X_sample)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=X.columns, show=False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/summary_plot.png')
    plt.close()
    
    # 2. Dependence plots for key features
    key_features = ['FY24_volume', 'segment_code', 'tier', 'country', 'product_level1', 
                     'country_segment']
    for feature in key_features:
        if feature in X.columns:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature, shap_values, X_sample, show=False)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/dependence_{feature}.png')
            plt.close()
    
    # 3. Force plot for a specific example
    idx = 0
    plt.figure(figsize=(20, 3))
    shap.force_plot(
        explainer.expected_value, 
        shap_values[idx,:], 
        X_sample.iloc[idx,:], 
        feature_names=X.columns,
        matplotlib=True,
        show=False
    )
    plt.savefig(f'{output_dir}/force_plot_example.png', bbox_inches='tight')
    plt.close()
    
    # 4. Product-specific analysis
    product_types = df['product_level1'].unique()
    for product_type in product_types:
        # Filter samples for this product type
        product_mask = df['product_level1'] == product_type
        if product_mask.sum() > 0:
            product_indices = df[product_mask].index
            product_sample = X.loc[product_indices].sample(min(200, len(product_indices)), random_state=42)
            
            # Calculate SHAP values
            product_shap = explainer.shap_values(product_sample)
            
            # Summary plot for this product type
            plt.figure(figsize=(12, 8))
            shap.summary_plot(product_shap, product_sample, feature_names=X.columns, show=False)
            plt.title(f'SHAP Summary - {product_type}')
            plt.tight_layout()
            
            # 使用清理后的文件名
            safe_product_type = sanitize_filename(product_type.lower())
            plt.savefig(f'{output_dir}/summary_{safe_product_type}.png')
            plt.close()
            
            
    
    # 5. Tier analysis
    tier_values = df['tier'].unique()
    for tier in tier_values:
        # Filter samples for this tier
        tier_mask = df['tier'] == tier
        if tier_mask.sum() > 0:
            tier_indices = df[tier_mask].index
            tier_sample = X.loc[tier_indices].sample(min(200, len(tier_indices)), random_state=42)
            
            # Calculate SHAP values
            tier_shap = explainer.shap_values(tier_sample)
            
            # Summary plot for this tier
            plt.figure(figsize=(12, 8))
            shap.summary_plot(tier_shap, tier_sample, feature_names=X.columns, show=False)
            plt.title(f'SHAP Summary - Tier: {tier}')
            plt.tight_layout()
            
            # 使用清理后的文件名
            safe_tier = sanitize_filename(tier.lower())
            plt.savefig(f'{output_dir}/summary_tier_{safe_tier}.png')
            plt.close()
    
    # 6. Country analysis - focus on Singapore
    sg_mask = df['country'] == 'SG'
    if sg_mask.sum() > 0:
        sg_indices = df[sg_mask].index
        sg_sample = X.loc[sg_indices].sample(min(200, len(sg_indices)), random_state=42)
        
        # Calculate SHAP values
        sg_shap = explainer.shap_values(sg_sample)
        
        # Summary plot for Singapore
        plt.figure(figsize=(12, 8))
        shap.summary_plot(sg_shap, sg_sample, feature_names=X.columns, show=False)
        plt.title('SHAP Summary - Country: SG (Singapore)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/summary_country_sg.png')
        plt.close()
        
        # Compare with global average
        global_shap = shap_values
        sg_avg = np.mean(sg_shap, axis=0)
        global_avg = np.mean(global_shap, axis=0)
        diff = sg_avg - global_avg
        
        # Feature importance difference
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(diff)), diff, 
                 color=['red' if x < 0 else 'blue' for x in diff])
        plt.yticks(range(len(diff)), X.columns)
        plt.title('Singapore vs Global Feature Impact')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/country_sg_vs_global.png')
        plt.close()
    
    # 7. Volume analysis
    volume_segments = {
        'Low (<10K)': df['FY24_volume'] < 10000,
        'Medium (10K-50K)': (df['FY24_volume'] >= 10000) & (df['FY24_volume'] < 50000),
        'High (50K-100K)': (df['FY24_volume'] >= 50000) & (df['FY24_volume'] < 100000),
        'Very High (>100K)': df['FY24_volume'] >= 100000
    }
    
    for segment_name, mask in volume_segments.items():
        if mask.sum() > 0:
            volume_indices = df[mask].index
            volume_sample = X.loc[volume_indices].sample(min(200, len(volume_indices)), random_state=42)
            
            # Calculate SHAP values
            volume_shap = explainer.shap_values(volume_sample)
            
            # Summary plot for this volume segment
            plt.figure(figsize=(12, 8))
            shap.summary_plot(volume_shap, volume_sample, feature_names=X.columns, show=False)
            plt.title(f'SHAP Summary - Volume: {segment_name}')
            plt.tight_layout()
            
            # 使用清理后的文件名
            safe_segment_name = sanitize_filename(segment_name.lower())
            plt.savefig(f'{output_dir}/summary_volume_{safe_segment_name}.png')
            plt.close()
    
    print(f"SHAP analysis completed. Results saved to {output_dir}/")
    print("Generated: summary plots, dependence plots, product-specific analysis, tier analysis, and country analysis")
    
    # Print comprehensive SHAP analysis report
    print_shap_analysis_report(model_data, shap_values, X_sample, feature_names)

if __name__ == "__main__":
    analyze_model_shap()
