"""
Quick analysis script to generate results without hyperparameter tuning.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from causal_hba1c.data.loader import DiabetesDataLoader
from causal_hba1c.data.preprocessor import DiabetesPreprocessor
from causal_hba1c.models.causal_models import CausalInferenceEngine
from causal_hba1c.visualization.plots import CausalVisualization
from causal_hba1c.utils.helpers import (
    setup_logging, save_results, calculate_summary_statistics,
    check_balance, create_age_groups, calculate_treatment_intensity,
    validate_data_quality, format_results_table, ensure_directory
)


def main():
    """Run a quick causal inference analysis without hyperparameter tuning."""
    
    # Setup
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)
    
    # Create output directories
    figures_dir = ensure_directory("figures")
    results_dir = ensure_directory("results")
    data_dir = ensure_directory("data/processed")
    
    logger.info("Starting quick causal inference analysis")
    
    # Load existing processed data if available
    processed_data_path = data_dir / "processed_diabetes_data.csv"
    if processed_data_path.exists():
        logger.info("Loading existing processed data")
        processed_data = pd.read_csv(processed_data_path)
    else:
        logger.info("Processing data from scratch")
        # Load and preprocess data
        loader = DiabetesDataLoader()
        raw_data = loader.load_data(remove_duplicates=True)
        
        preprocessor = DiabetesPreprocessor()
        processed_data = preprocessor.preprocess(raw_data)
        
        # Add age groups and treatment intensity
        processed_data = create_age_groups(processed_data)
        processed_data = calculate_treatment_intensity(processed_data)
        
        # Save processed data
        processed_data.to_csv(processed_data_path, index=False)
    
    logger.info(f"Dataset shape: {processed_data.shape}")
    
    # Initialize causal inference engine
    causal_engine = CausalInferenceEngine(n_folds=3, random_state=42)  # Reduced folds for speed
    
    # Prepare data for analysis
    dml_data = causal_engine.prepare_data(processed_data)
    
    # Estimate treatment effects without hyperparameter tuning
    logger.info("Estimating treatment effects (no hyperparameter tuning)")
    treatment_effects = causal_engine.estimate_treatment_effects(
        dml_data, 
        methods=['logistic', 'random_forest', 'xgboost'],
        tune_hyperparameters=False
    )
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("TREATMENT EFFECT RESULTS")
    logger.info("="*50)
    
    for method, estimate in treatment_effects.items():
        logger.info(f"\n{method.upper()}:")
        logger.info(f"  Coefficient: {estimate.coefficient:.6f}")
        logger.info(f"  Std Error: {estimate.std_error:.6f}")
        logger.info(f"  95% CI: [{estimate.ci_lower:.6f}, {estimate.ci_upper:.6f}]")
        logger.info(f"  P-value: {estimate.p_value:.6f}")
        logger.info(f"  Significant: {'Yes' if estimate.is_significant else 'No'}")
    
    # Create treatment effects visualization
    visualizer = CausalVisualization()
    try:
        visualizer.plot_treatment_effects(
            treatment_effects, 
            save_path=figures_dir / "treatment_effects.png"
        )
        logger.info("Treatment effects plot saved")
    except Exception as e:
        logger.error(f"Error creating treatment effects plot: {e}")
    
    # Model performance evaluation
    try:
        logger.info("Evaluating model performance")
        performance = causal_engine.evaluate_learner_performance()
        
        logger.info("\nMODEL PERFORMANCE (Log-Loss):")
        for method, metrics in performance.items():
            if 'error' not in metrics:
                logger.info(f"{method}: {metrics}")
        
        # Create model performance plot
        visualizer.plot_model_performance(
            performance,
            save_path=figures_dir / "model_performance.png"
        )
        logger.info("Model performance plot saved")
        
    except Exception as e:
        logger.error(f"Error in performance evaluation: {e}")
    
    # Heterogeneous effects analysis
    try:
        if 'age_group' in processed_data.columns:
            logger.info("Analyzing heterogeneous effects by age group")
            hetero_effects = causal_engine.analyze_heterogeneous_effects(
                processed_data, 
                ['age_group'],
                method='random_forest'
            )
            
            # Create heterogeneous effects plot
            if 'age_group' in hetero_effects and hetero_effects['age_group']:
                visualizer.plot_heterogeneous_effects(
                    {'age_group': hetero_effects['age_group']},
                    save_path=figures_dir / "heterogeneous_effects.png"
                )
                logger.info("Heterogeneous effects plot saved")
                
                # Print heterogeneous results
                logger.info("\nHETEROGENEOUS EFFECTS BY AGE GROUP:")
                for group, effect in hetero_effects['age_group'].items():
                    logger.info(f"  {group}: {effect.coefficient:.6f} (p={effect.p_value:.3f})")
        
    except Exception as e:
        logger.error(f"Error in heterogeneous effects analysis: {e}")
    
    # Save results
    results = {
        'treatment_effects': {k: {
            'coefficient': v.coefficient,
            'std_error': v.std_error,
            'ci_lower': v.ci_lower,
            'ci_upper': v.ci_upper,
            'p_value': v.p_value,
            'method': v.method
        } for k, v in treatment_effects.items()},
        'dataset_info': {
            'n_observations': len(processed_data),
            'n_features': len(processed_data.columns),
            'treatment_rate': processed_data['A1C_tested'].mean(),
            'outcome_rate': processed_data['readmitted'].mean()
        }
    }
    
    save_results(results, results_dir / "causal_analysis_results.json")
    logger.info("Results saved to JSON file")
    
    logger.info("Quick analysis completed!")


if __name__ == "__main__":
    main()