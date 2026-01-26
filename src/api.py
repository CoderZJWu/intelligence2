"""
FastAPI implementation for the pricing prediction service.
Provides REST API for real-time pricing predictions.
"""

from fastapi import FastAPI, HTTPException, Query
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from .pricing_rules import validate_price_structure, apply_tier_discount, get_tier_comparison

app = FastAPI(
    title="GPBS Pricing Simulator API",
    description="AI-powered pricing prediction for Global Pricing & Billing System",
    version="1.1.0"
)

# Load model and metadata
model_data = joblib.load('models/lgbm_model.pkl')
model = model_data['model']
label_encoders = model_data['label_encoders']
cat_features = model_data['cat_features']
most_frequent_classes = model_data.get('most_frequent_classes', {})
category_mappings = model_data.get('category_mappings', {})
segment_mapping = model_data.get('segment_mapping', {})  # Get segment mapping
product_combinations = joblib.load('models/product_combinations.pkl')
feature_names = model_data['feature_names']

class PricingRequest(BaseModel):
    country: str
    segment_code: int
    tier: str
    volume: float
    currency: str = "USD"
    include_comparisons: bool = False

class PricingResponse(BaseModel):
    pricing: Dict[str, List[Dict[str, Dict[str, str]]]]
    metadata: Dict[str, Any] = {}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "GPBS Pricing Simulator API",
        "version": "1.1.0",
        "endpoints": [
            "/predict - POST endpoint for pricing predictions",
            "/health - Health check endpoint",
            "/explain?feature=[feature] - SHAP explanation for specific feature"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True}

@app.get("/explain")
async def explain_feature(feature: str = Query(..., description="Feature to explain")):
    """Get explanation for how a specific feature affects pricing"""
    if feature not in feature_names:
        raise HTTPException(status_code=400, detail=f"Feature '{feature}' not found in model features")
    
    # Get feature importance
    feature_idx = feature_names.index(feature)
    importance = model.feature_importance()[feature_idx]
    
    # Create explanation based on feature type
    explanation = {}
    
    if feature == 'service_level_factor':
        explanation = {
            "feature": "service_level_factor",
            "description": "Service level premium factor (STD=1.0, PRM=1.2, VIP=1.45, ELT=1.75)",
            "impact": "Higher service levels increase price proportionally",
            "typical_impact": "+20% to +75% based on service level"
        }
    elif feature == 'tier':
        explanation = {
            "feature": "tier",
            "description": "Customer tier (Gold, Platinum, Portfolio, Silver)",
            "impact": "Higher tiers get price discounts",
            "tier_discounts": {
                "Gold": "-10% to -15%",
                "Platinum": "-5% to -10%",
                "Portfolio": "-2% to -5%",
                "Silver": "Base price"
            }
        }
    elif feature == 'country':
        explanation = {
            "feature": "country",
            "description": "Customer country",
            "impact": "Country-specific pricing adjustments",
            "country_examples": {
                "SG": "-2% to -8% (Singapore market discounts)",
                "US": "+5% (US market premium)"
            }
        }
    elif feature == 'FY24_volume':
        explanation = {
            "feature": "FY24_volume",
            "description": "Expected transaction volume",
            "impact": "Higher volume leads to volume discounts",
            "discount_curve": "Discount = 1 / (1 + log10(volume/1000))"
        }
    else:
        explanation = {
            "feature": feature,
            "description": f"Feature '{feature}' used in pricing model",
            "importance": float(importance)
        }
    
    return explanation

@app.post("/predict", response_model=PricingResponse)
async def predict_pricing(request: PricingRequest):
    """Predict pricing for all products based on input parameters"""
    try:
        # Create a dataframe for each product combination
        prediction_data = []
        
        # Get segment name from mapping
        segment_name = segment_mapping.get(request.segment_code, "Unknown Segment")
        
        for _, row in product_combinations.iterrows():
            data = {
                'country': request.country,
                'segment_code': request.segment_code,
                'tier': request.tier,
                'product_level1': row['product_level1'],
                'product_level2': row['product_level2'],
                'product_code': row['product_code'],
                'service_level': row['service_level'],  # NEW
                'charge_currency': request.currency,
                'computation_method': 'Fixed flat',  # Default value
                'FY24_volume': request.volume,
                'volume_per_transaction': request.volume / 100,  # Simplified estimate
                'revenue_growth': 0.05,  # Simplified estimate
                'volume_growth': 0.03,    # Simplified estimate
                'avg_cp_at_country': 0.5,  # Simplified value
                'avg_cp_at_seg': 0.5,      # Simplified value
                'avg_cp_at_tier': 0.5,     # Simplified value
                'avg_cp_at_wa_volume': 0.5, # Simplified value
                'service_level_factor': {
                    'STD': 1.0,
                    'PRM': 1.2,
                    'VIP': 1.45,
                    'ELT': 1.75
                }.get(row['service_level'], 1.0),  # NEW
                'is_premium_service': 1 if row['service_level'] != 'STD' else 0,  # NEW
                'country_segment': f"{request.country}_{segment_name.replace(' ', '_')}"  # FIXED
            }
            prediction_data.append(data)
        
        df = pd.DataFrame(prediction_data)
        
        # Use the exact feature order from training
        df = df[[f for f in feature_names if f in df.columns]]
        
        # Encode categorical features using explicit mappings
        for feature in cat_features:
            if feature in df.columns:
                if feature in category_mappings:
                    # Get default value (most frequent class encoded value)
                    default_value = 0
                    if feature in most_frequent_classes:
                        mf_class = str(most_frequent_classes[feature])
                        if mf_class in category_mappings[feature]:
                            default_value = category_mappings[feature][mf_class]
                    
                    # Convert to string for mapping
                    df[feature] = df[feature].astype(str)
                    
                    # Apply mapping with default value for unknown categories
                    df[feature] = df[feature].map(category_mappings[feature]).fillna(default_value)
                else:
                    # Fallback for segment_code
                    if feature == 'segment_code':
                        df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
                    else:
                        df[feature] = df[feature].astype('category').cat.codes
        
        # Predict
        predictions = model.predict(df)
        
        # Format results in hierarchical structure
        results = {}
        
        for i, pred in enumerate(predictions):
            p1 = prediction_data[i]['product_level1']
            p2 = prediction_data[i]['product_level2']
            p_code = prediction_data[i]['product_code']
            
            if p1 not in results:
                results[p1] = []
            
            # Find if this product_level2 already exists in the list
            p2_found = False
            for item in results[p1]:
                if p2 in item:
                    item[p2][p_code] = f"{pred:.4f}"
                    p2_found = True
                    break
            
            if not p2_found:
                results[p1].append({
                    p2: {p_code: f"{pred:.4f}"}
                })
        
        # Apply business rules validation
        validated_results = validate_price_structure(results)
        
        # Apply tier-based pricing
        final_results = apply_tier_discount(validated_results, request.tier)
        
        # Prepare metadata
        metadata = {
            "request": request.dict(),
            "model_version": "1.1",
            "timestamp": pd.Timestamp.now().isoformat(),
            "product_count": sum(len(items) for items in final_results.values())
        }
        
        # Add tier comparisons if requested
        if request.include_comparisons:
            metadata["tier_comparisons"] = get_tier_comparison(final_results, request.tier)
        
        return PricingResponse(
            pricing=final_results,
            metadata=metadata
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
