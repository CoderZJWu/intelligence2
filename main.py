"""
Main entry point for the GPBS-Intelligence application.
"""
import uvicorn
from src.api import app
import sys
import os
import streamlit as st

def run_full_pipeline():
    """Run the complete model training, evaluation and analysis pipeline"""
    print("Starting GPBS-Intelligence pipeline...\n")
    
    print("="*50)
    print("STEP 1: Generating simulated data")
    print("="*50)
    from src.data_generator import generate_simulated_data
    generate_simulated_data(n_samples=10000)
    
    print("\n" + "="*50)
    print("STEP 2: Training pricing model")
    print("="*50)
    from src.model import train_pricing_model
    train_pricing_model()
    
    print("\n" + "="*50)
    print("STEP 3: Performing SHAP analysis")
    print("="*50)
    from src.shap_analysis import analyze_model_shap
    analyze_model_shap()
    
    print("\n" + "="*50)
    print("GPBS-Intelligence pipeline completed successfully!")
    print("Reports saved to 'reports/' directory")
    print("To start the API server, run: python main.py")
    print("To start the pricing dashboard, run: streamlit run pricing_dashboard.py")
    print("="*50)

def run_pricing_dashboard():
    """Run the pricing dashboard using Streamlit"""
    os.system("streamlit run pricing_dashboard.py")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pipeline":
        run_full_pipeline()
    elif len(sys.argv) > 1 and sys.argv[1] == "dashboard":
        run_pricing_dashboard()
    else:
        print("Starting GPBS-Intelligence API server...")
        print("Visit http://localhost:8000/docs for API documentation")
        print("To run the pricing dashboard, use: python main.py dashboard")
        print("To run the full pipeline, use: python main.py pipeline")
        uvicorn.run(app, host="0.0.0.0", port=8000)
