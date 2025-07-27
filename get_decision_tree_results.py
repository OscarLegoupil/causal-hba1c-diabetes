"""
Get decision tree results by using a simple approach.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import doubleml as dml

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from causal_hba1c.utils.helpers import setup_logging, ensure_directory

def main():
    """Get decision tree results manually."""
    
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)
    
    # Load processed data
    data_dir = ensure_directory("data/processed")
    processed_data_path = data_dir / "processed_diabetes_data.csv"
    processed_data = pd.read_csv(processed_data_path)
    
    logger.info(f"Dataset shape: {processed_data.shape}")
    
    # Prepare data
    exclude_cols = ['A1C_tested', 'readmitted', 'A1Cresult_ord', 'age_group', 'treatment_intensity']
    
    # Get numeric columns only
    x_cols = []
    for col in processed_data.columns:
        if col not in exclude_cols:
            if processed_data[col].dtype in ['int64', 'float64', 'bool']:
                x_cols.append(col)
    
    X = processed_data[x_cols]
    y = processed_data['readmitted']
    d = processed_data['A1C_tested']
    
    logger.info(f"Using {len(x_cols)} covariates")
    
    # Create DoubleML data object
    dml_data = dml.DoubleMLData(
        processed_data,
        y_col='readmitted',
        d_cols='A1C_tested',
        x_cols=x_cols
    )
    
    # Try decision tree with proper settings for probability prediction
    from sklearn.tree import DecisionTreeClassifier
    
    # Use deeper trees that can output probabilities
    dt_g = DecisionTreeClassifier(
        random_state=42, 
        min_samples_leaf=50,  # Larger leaf size
        max_depth=10,         # Limit depth
        min_samples_split=100  # Larger split requirement
    )
    
    dt_m = DecisionTreeClassifier(
        random_state=42,
        min_samples_leaf=50,
        max_depth=10,
        min_samples_split=100
    )
    
    try:
        # Create Double ML IRM model with decision trees
        dml_model = dml.DoubleMLIRM(
            dml_data,
            ml_g=dt_g,
            ml_m=dt_m,
            n_folds=3
        )
        
        # Fit the model
        logger.info("Fitting decision tree model...")
        dml_model.fit(store_predictions=True)
        
        # Extract results
        summary = dml_model.summary
        first_row = summary.iloc[0]
        
        logger.info("DECISION TREE RESULTS:")
        logger.info(f"  Coefficient: {first_row['coef']:.6f}")
        logger.info(f"  Std Error: {first_row['std err']:.6f}")
        logger.info(f"  95% CI: [{first_row['2.5 %']:.6f}, {first_row['97.5 %']:.6f}]")
        logger.info(f"  P-value: {first_row['P>|t|']:.6f}")
        logger.info(f"  Significant: {'Yes' if first_row['P>|t|'] < 0.05 else 'No'}")
        
    except Exception as e:
        logger.error(f"Decision tree failed: {e}")
        logger.info("Trying with Random Forest instead...")
        
        # Fallback to Random Forest
        rf_g = RandomForestClassifier(random_state=42, n_estimators=50)
        rf_m = RandomForestClassifier(random_state=42, n_estimators=50)
        
        dml_model = dml.DoubleMLIRM(
            dml_data,
            ml_g=rf_g,
            ml_m=rf_m,
            n_folds=3
        )
        
        dml_model.fit(store_predictions=True)
        summary = dml_model.summary
        first_row = summary.iloc[0]
        
        logger.info("RANDOM FOREST RESULTS (as proxy for decision tree):")
        logger.info(f"  Coefficient: {first_row['coef']:.6f}")
        logger.info(f"  Std Error: {first_row['std err']:.6f}")
        logger.info(f"  95% CI: [{first_row['2.5 %']:.6f}, {first_row['97.5 %']:.6f}]")
        logger.info(f"  P-value: {first_row['P>|t|']:.6f}")
        logger.info(f"  Significant: {'Yes' if first_row['P>|t|'] < 0.05 else 'No'}")

if __name__ == "__main__":
    main()