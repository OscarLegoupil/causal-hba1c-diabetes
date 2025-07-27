"""
Main analysis script for causal inference of HbA1c testing on hospital readmissions.

This script implements a comprehensive causal inference pipeline using Double Machine Learning
to estimate the effect of HbA1c testing on 30-day hospital readmission rates for diabetic patients.
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
    """Run the complete causal inference analysis pipeline."""
    
    # Setup
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)
    
    # Create output directories
    figures_dir = ensure_directory("figures")
    results_dir = ensure_directory("results")
    data_dir = ensure_directory("data/processed")
    
    logger.info("Starting causal inference analysis of HbA1c testing impact on readmissions")
    
    # Step 1: Load and preprocess data
    logger.info("Step 1: Loading and preprocessing data")
    
    # Load data
    loader = DiabetesDataLoader()
    raw_data = loader.load_data(remove_duplicates=True)
    
    # Data quality validation
    quality_metrics = validate_data_quality(raw_data)
    logger.info(f"Data quality: {quality_metrics['n_observations']} observations, "
               f"{quality_metrics['n_features']} features")
    
    # Preprocess data
    preprocessor = DiabetesPreprocessor()
    processed_data = preprocessor.preprocess(raw_data)
    
    # Add age groups and treatment intensity
    processed_data = create_age_groups(processed_data)
    processed_data = calculate_treatment_intensity(processed_data)
    
    # Save processed data
    processed_data.to_csv(data_dir / "processed_diabetes_data.csv", index=False)
    logger.info("Processed data saved")
    
    # Step 2: Exploratory Data Analysis
    logger.info("Step 2: Exploratory data analysis and visualization")
    
    visualizer = CausalVisualization()
    
    # Data overview
    visualizer.plot_data_overview(processed_data, save_path=figures_dir / "data_overview.png")
    
    # Correlation analysis
    key_variables = [
        'gender', 'age', 'time_in_hospital', 'num_lab_procedures',
        'num_medications', 'A1C_tested', 'diabetesMed', 'readmitted',
        'emergency', 'diag_Diabetes'
    ]
    available_vars = [var for var in key_variables if var in processed_data.columns]
    visualizer.create_correlation_heatmap(
        processed_data, available_vars, 
        save_path=figures_dir / "correlation_matrix.png"
    )
    
    # Causal DAG
    visualizer.plot_causal_dag(save_path=figures_dir / "causal_dag.png")
    
    # Subgroup analysis plots
    if 'age_group' in processed_data.columns:
        visualizer.plot_readmission_by_subgroups(
            processed_data, 'age_group',
            save_path=figures_dir / "readmission_by_age_group.png"
        )
    
    # Step 3: Covariate Balance Analysis
    logger.info("Step 3: Checking covariate balance")
    
    feature_groups = preprocessor.get_feature_groups(processed_data)
    all_covariates = []
    for group_vars in feature_groups.values():
        all_covariates.extend([var for var in group_vars if var in processed_data.columns])
    
    # Remove treatment and outcome variables from covariates
    covariates = [var for var in all_covariates 
                 if var not in ['A1C_tested', 'readmitted', 'A1Cresult_ord']]
    
    balance_stats = check_balance(processed_data, 'A1C_tested', covariates)
    
    # Identify imbalanced covariates (|SMD| > 0.1)
    imbalanced = balance_stats[balance_stats['standardized_mean_diff'].abs() > 0.1]
    logger.info(f"Found {len(imbalanced)} imbalanced covariates (|SMD| > 0.1)")
    
    # Step 4: Causal Inference Analysis
    logger.info("Step 4: Causal inference using Double Machine Learning")
    
    # Initialize causal inference engine
    causal_engine = CausalInferenceEngine(n_folds=5, random_state=42)
    
    # Prepare data for causal analysis
    dml_data = causal_engine.prepare_data(processed_data)
    
    # Estimate treatment effects using multiple methods
    treatment_effects = causal_engine.estimate_treatment_effects(
        dml_data, 
        methods=['logistic', 'random_forest', 'decision_tree', 'xgboost'],
        tune_hyperparameters=True
    )
    
    # Evaluate model performance
    model_performance = causal_engine.evaluate_learner_performance()
    
    # Visualize treatment effects
    visualizer.plot_treatment_effects(
        treatment_effects,
        save_path=figures_dir / "treatment_effects.png"
    )
    
    # Model performance plot
    visualizer.plot_model_performance(
        model_performance,
        save_path=figures_dir / "model_performance.png"
    )
    
    # Step 5: Heterogeneous Treatment Effects
    logger.info("Step 5: Analyzing heterogeneous treatment effects")
    
    subgroup_vars = ['age_group']
    if 'emergency' in processed_data.columns:
        subgroup_vars.append('emergency')
    
    heterogeneous_effects = causal_engine.analyze_heterogeneous_effects(
        processed_data, 
        subgroup_vars,
        method='random_forest'
    )
    
    # Visualize heterogeneous effects
    if heterogeneous_effects:
        visualizer.plot_heterogeneous_effects(
            heterogeneous_effects,
            save_path=figures_dir / "heterogeneous_effects.png"
        )
    
    # Step 6: Sensitivity Analysis and Placebo Tests
    logger.info("Step 6: Sensitivity analysis and placebo tests")
    
    # Placebo test using gender
    placebo_result = causal_engine.run_placebo_test(processed_data, placebo_treatment='gender')
    logger.info(f"Placebo test result: coefficient={placebo_result.coefficient:.6f}, "
               f"p-value={placebo_result.p_value:.4f}")
    
    # Step 7: Results Summary and Reporting
    logger.info("Step 7: Generating results summary")
    
    # Compile all results
    results = {
        'data_summary': {
            'n_observations': len(processed_data),
            'n_features': len(processed_data.columns),
            'treatment_rate': processed_data['A1C_tested'].mean(),
            'readmission_rate': processed_data['readmitted'].mean()
        },
        'treatment_effects': {
            method: {
                'coefficient': est.coefficient,
                'std_error': est.std_error,
                'ci_lower': est.ci_lower,
                'ci_upper': est.ci_upper,
                'p_value': est.p_value,
                'significant': est.is_significant
            } for method, est in treatment_effects.items()
        },
        'model_performance': model_performance,
        'heterogeneous_effects': {
            var: {
                subgroup: {
                    'coefficient': est.coefficient,
                    'p_value': est.p_value,
                    'significant': est.is_significant
                } for subgroup, est in effects.items()
            } for var, effects in heterogeneous_effects.items()
        } if heterogeneous_effects else {},
        'placebo_test': {
            'coefficient': placebo_result.coefficient,
            'p_value': placebo_result.p_value,
            'significant': placebo_result.is_significant
        },
        'balance_analysis': {
            'n_imbalanced_covariates': len(imbalanced),
            'max_imbalance': balance_stats['standardized_mean_diff'].abs().max()
        }
    }
    
    # Save results
    save_results(results, results_dir / "causal_analysis_results.json")
    
    # Print summary
    print("\n" + "="*80)
    print("CAUSAL INFERENCE ANALYSIS RESULTS")
    print("="*80)
    
    print(format_results_table(treatment_effects, "Treatment Effect Estimates"))
    
    print(f"\nPlacebo Test (Gender as Treatment):")
    print(f"Coefficient: {placebo_result.coefficient:.6f}, P-value: {placebo_result.p_value:.4f}")
    print(f"Significant: {'Yes' if placebo_result.is_significant else 'No'}")
    
    if heterogeneous_effects:
        print(f"\nHeterogeneous Effects Summary:")
        for var, effects in heterogeneous_effects.items():
            print(f"\n{var.upper()}:")
            for subgroup, est in effects.items():
                sig_marker = "*" if est.is_significant else ""
                print(f"  {subgroup}: {est.coefficient:.6f} (p={est.p_value:.4f}){sig_marker}")
    
    print(f"\nData Summary:")
    print(f"- Total observations: {len(processed_data):,}")
    print(f"- A1C testing rate: {processed_data['A1C_tested'].mean():.1%}")
    print(f"- 30-day readmission rate: {processed_data['readmitted'].mean():.1%}")
    
    logger.info("Analysis complete! Check the figures/ and results/ directories for outputs.")
    print(f"\nAnalysis complete! Outputs saved to:")
    print(f"- Figures: {figures_dir}")
    print(f"- Results: {results_dir}")


if __name__ == "__main__":
    main()