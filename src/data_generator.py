"""
Generates simulated banking pricing data for GPBS system.
Creates realistic dataset with all required features for pricing prediction.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_simulated_data(n_samples=10000):
    """Generate simulated banking pricing data"""
    
    # Seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define possible values for categorical features
    countries = ['SG', 'US', 'GB', 'DE', 'HK', 'JP', 'AU']
    segments = {
        'Banks': 600,
        'Global Corporates': 500,
        'Middle Market': 300,
        'Traders & Buyers': 460
    }
    tiers = ['Gold', 'Platinum', 'Portfolio', 'Silver']
    product_level1 = ['Payments', 'Clearing', 'Collections', 'Liquidity']
    product_level2 = {
        'Payments': ['ACH', 'BT', 'CASH', 'CO', 'OTT', 'Pay'],
        'Clearing': ['Clearing'],
        'Collections': ['CASH', 'DDA','ITT'],
        'Liquidity': ['Setup & Maintenance']
    }
    charge_currencies = ['SGD', 'USD']
    computation_methods = ['Fixed flat', 'flat absolute']
    
    # Generate product codes with meaningful variations
    product_codes = []
    for p1 in product_level1:
        for p2 in product_level2[p1]:
            # Create variations in product codes that affect pricing
            for i in range(1, 5):
                # Add suffixes that indicate different service levels
                service_levels = ['STD', 'PRM', 'VIP', 'ELT']
                product_codes.append(f"CAP{p1[:3].upper()}{p2[:3].upper()}{service_levels[i-1]}B{i}")
    
    # Generate CRM IDs and customer names
    crm_ids = [f"CRM{str(i).zfill(6)}" for i in range(1, n_samples+1)]
    customer_names = [f"Customer_{str(i).zfill(4)}" for i in range(1, n_samples+1)]
    
    # Generate segment names and codes
    segment_names = []
    segment_codes = []
    for _ in range(n_samples):
        name = random.choice(list(segments.keys()))
        segment_names.append(name)
        segment_codes.append(segments[name])
    
    # Generate tier
    tier_list = [random.choice(tiers) for _ in range(n_samples)]
    
    # Generate product level 1 and 2
    p1_list = [random.choice(product_level1) for _ in range(n_samples)]
    p2_list = [random.choice(product_level2[p1]) for p1 in p1_list]
    
    # Generate product codes based on product levels with pricing variations
    product_code_list = []
    for i in range(n_samples):
        # Filter codes matching product levels
        matching_codes = [code for code in product_codes 
                         if code.startswith(f"CAP{p1_list[i][:3].upper()}{p2_list[i][:3].upper()}")]
        
        if matching_codes:
            # Assign different prices based on service level
            service_level_prices = {
                'STD': 1.0,   # Standard
                'PRM': 1.15,  # Premium
                'VIP': 1.3,   # VIP
                'ELT': 1.5    # Elite
            }
            
            # Select code with weighted probability (more standard, fewer elite)
            weights = [0.5, 0.3, 0.15, 0.05]  # STD, PRM, VIP, ELT
            selected_code = random.choices(matching_codes, weights=weights, k=1)[0]
            
            # Store the service level for pricing logic
            service_level = selected_code[9:12]  # Extract service level from code
            product_code_list.append((selected_code, service_level))
        else:
            product_code_list.append(("UNKNOWN", "STD"))
    
    # Extract product codes and service levels
    product_codes_only = [item[0] for item in product_code_list]
    service_levels = [item[1] for item in product_code_list]
    
    # Generate charge currency
    charge_currency_list = [random.choice(charge_currencies) for _ in range(n_samples)]
    
    # Generate computation method
    computation_method_list = [random.choice(computation_methods) for _ in range(n_samples)]
    
    # Generate volume and fee data with realistic relationships
    fy23_volumes = np.random.lognormal(mean=8, sigma=1.5, size=n_samples).astype(int)
    fy24_volumes = np.random.lognormal(mean=8.2, sigma=1.4, size=n_samples).astype(int)
    
    # Generate ecp_usd with realistic pricing logic
    ecp_usd_list = []
    for i in range(n_samples):
        base_price = 1.0
        
        # Tier adjustment
        if tier_list[i] == 'Gold':
            base_price *= 0.85
        elif tier_list[i] == 'Platinum':
            base_price *= 0.9
        elif tier_list[i] == 'Silver':
            base_price *= 1.15
        
        # Segment adjustment
        if segment_names[i] == 'Banks':
            base_price *= 0.95
        elif segment_names[i] == 'Traders & Buyers':
            base_price *= 1.1
        
        # Product adjustment
        if p1_list[i] == 'Payments':
            base_price *= 1.0
        elif p1_list[i] == 'Clearing':
            base_price *= 1.25
        elif p1_list[i] == 'Liquidity':
            base_price *= 0.8
            
        
        # Volume discount
        volume_factor = 1 / (1 + np.log10(fy24_volumes[i] / 1000))
        
        # Country adjustment with Singapore-specific pricing
        country_idx = segment_codes[i] % len(countries)
        country = countries[country_idx]
        
        if country == 'SG':
            # Singapore-specific pricing: slightly lower for some products
            if p1_list[i] in ['Payments', 'Clearing']:
                base_price *= 0.92
            else:
                base_price *= 0.98
        elif country == 'US':
            base_price *= 1.05
        
        # Add noise with controlled variation
        price = base_price * volume_factor * (0.98 + random.random() * 0.04)
        ecp_usd_list.append(round(price, 4))
    
    # Generate esp_usd (similar but with slight variation)
    esp_usd_list = [round(price * (0.98 + random.random() * 0.04), 4) for price in ecp_usd_list]
    
    # Generate FY23 and FY24 fees
    fy23_fees = [round(ecp_usd_list[i] * fy23_volumes[i], 2) for i in range(n_samples)]
    fy24_fees = [round(ecp_usd_list[i] * fy24_volumes[i], 2) for i in range(n_samples)]
    
    # Generate revenue bins with the updated values
    cash_revenue_bins = ['>2M', '0-10K', '10K-200K', '200K-600K', '600K-2M']
    cib_revenue_bins = ['>5M', '0-100K', '100K-500K', '500K-2M', '2M-5M']
    
    # Generate avg cp features
    avg_cp_country = {c: round(random.uniform(0.4, 0.8), 4) for c in countries}
    avg_cp_cash_bin = {b: round(random.uniform(0.3, 0.9), 4) for b in cash_revenue_bins}
    avg_cp_cib_bin = {b: round(random.uniform(0.3, 0.9), 4) for b in cib_revenue_bins}
    avg_cp_seg = {s: round(random.uniform(0.35, 0.85), 4) for s in segments.keys()}
    avg_cp_tier = {t: round(random.uniform(0.3, 0.9), 4) for t in tiers}
    
    # Create the DataFrame
    data = {
        'country': [countries[segment_codes[i] % len(countries)] for i in range(n_samples)],
        'crm_id': crm_ids,
        'customer_name': customer_names,
        'segment_name': segment_names,
        'segment_code': segment_codes,
        'tier': tier_list,
        'product_level1': p1_list,
        'product_level2': p2_list,
        'product_code': product_codes_only,
        'service_level': service_levels,  # NEW COLUMN FOR SERVICE LEVEL
        'charge_code': [f"CHG{str(i).zfill(4)}" for i in range(1, n_samples+1)],
        'charge_currency': charge_currency_list,
        'computation_method': computation_method_list,
        'ecp_usd': ecp_usd_list,
        'esp_usd': esp_usd_list,
        'FY23_fee_in_usd': fy23_fees,
        'FY23_volume': fy23_volumes,
        'FY24_fee_in_usd': fy24_fees,
        'FY24_volume': fy24_volumes,
        'Cash_revenue_bin': [random.choice(cash_revenue_bins) for _ in range(n_samples)],
        'CIB_revenue_bin': [random.choice(cib_revenue_bins) for _ in range(n_samples)],
        'avg_cp_at_country': [avg_cp_country[c] for c in [countries[segment_codes[i] % len(countries)] for i in range(n_samples)]],
        'avg_cp_at_cash_bin': [avg_cp_cash_bin[b] for b in [random.choice(cash_revenue_bins) for _ in range(n_samples)]],
        'avg_cp_at_cib_rev_bin': [avg_cp_cib_bin[b] for b in [random.choice(cib_revenue_bins) for _ in range(n_samples)]],
        'avg_cp_at_seg': [avg_cp_seg[s] for s in segment_names],
        'avg_cp_at_tier': [avg_cp_tier[t] for t in tier_list],
        'avg_cp_at_wa_volume': [round(random.uniform(0.35, 0.85), 4) for _ in range(n_samples)]
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('data/simulated_data.csv', index=False)
    print(f"Generated {n_samples} simulated data records")
    print(f"Data saved to data/simulated_data.csv")
    print("\nFirst 5 rows of generated data:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    generate_simulated_data(n_samples=10000)
