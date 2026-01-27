# GPBS-Intelligence - AI Pricing Simulator

AI-powered pricing simulator for Global Pricing & Billing System (GPBS) that predicts optimal pricing based on historical data.

## Features
- Generates realistic simulated banking pricing data
- Feature engineering for banking pricing context
- LightGBM model for accurate pricing predictions
- SHAP analysis for model interpretability
- REST API for real-time pricing predictions

# Create virtual environment
python -m venv venv
# source venv/bin/activate  # Linux/Mac
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate Simulated Data
python src/data_generator.py

# Train the Model
python src/model.py

# Run SHAP Analysis (Optional)
python src/shap_analysis.py

# Start the API Server
uvicorn main:app --reload --port 8000

# Make API Request

curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "country": "SG",
  "segment_code": 600,
  "tier": "Gold",
  "volume": 5000,
  "currency": "USD"
}'

# View API Documentation
Visit http://localhost:8000/docs in your browser for interactive API documentation.

# Simple Demo
pip install streamlit lightgbm pandas numpy matplotlib seaborn

streamlit run simple_demo.py