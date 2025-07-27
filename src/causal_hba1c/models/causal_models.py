"""
Causal inference models using Double Machine Learning framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass

import doubleml as dml
from doubleml import DoubleMLData
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import log_loss, balanced_accuracy_score
from sklearn.model_selection import ParameterGrid


logger = logging.getLogger(__name__)


@dataclass
class CausalEstimate:
    """Container for causal effect estimates."""
    coefficient: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    method: str
    
    @property
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if effect is statistically significant."""
        return self.p_value < alpha


class CausalInferenceEngine:
    """
    Implements Double Machine Learning for causal inference using Interactive Regression Model (IRM).
    """
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        """
        Initialize the causal inference engine.
        
        Args:
            n_folds: Number of folds for cross-fitting
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        treatment_col: str = 'A1C_tested',
        outcome_col: str = 'readmitted',
        exclude_cols: Optional[List[str]] = None
    ) -> DoubleMLData:
        """
        Prepare data for Double ML analysis.
        
        Args:
            df: Preprocessed dataset
            treatment_col: Name of treatment variable
            outcome_col: Name of outcome variable
            exclude_cols: Additional columns to exclude from covariates
            
        Returns:
            DoubleMLData object ready for analysis
        """
        if exclude_cols is None:
            exclude_cols = []
        
        # Define columns to exclude from covariates
        default_exclude = [treatment_col, outcome_col, 'A1Cresult_ord']
        all_exclude = default_exclude + exclude_cols
        
        # Get covariate columns
        x_cols = [col for col in df.columns if col not in all_exclude]
        
        logger.info(f"Prepared data with {len(x_cols)} covariates, treatment: {treatment_col}, outcome: {outcome_col}")
        
        return DoubleMLData(
            df,
            y_col=outcome_col,
            d_cols=treatment_col,
            x_cols=x_cols
        )
    
    def _get_base_learners(self) -> Dict[str, Dict[str, Any]]:
        """Get base machine learning learners for nuisance estimation."""
        return {
            'logistic': {
                'ml_g': LogisticRegression(penalty=None, solver='newton-cg', max_iter=1000),
                'ml_m': LogisticRegression(penalty=None, solver='newton-cg', max_iter=1000)
            },
            'random_forest': {
                'ml_g': RandomForestClassifier(random_state=self.random_state),
                'ml_m': RandomForestClassifier(random_state=self.random_state)
            },
            'decision_tree': {
                'ml_g': DecisionTreeClassifier(random_state=self.random_state),
                'ml_m': DecisionTreeClassifier(random_state=self.random_state)
            },
            'xgboost': {
                'ml_g': XGBClassifier(random_state=self.random_state, objective="binary:logistic", 
                                     eval_metric="logloss", n_jobs=1),
                'ml_m': XGBClassifier(random_state=self.random_state, objective="binary:logistic", 
                                     eval_metric="logloss", n_jobs=1)
            }
        }
    
    def _get_hyperparameter_grids(self) -> Dict[str, Dict[str, Any]]:
        """Get hyperparameter grids for model tuning."""
        return {
            'random_forest': {
                'ml_g': {
                    'n_estimators': [100, 300],
                    'max_depth': [6, 10],
                    'min_samples_leaf': [5, 10],
                    'max_features': ['sqrt', 10]
                },
                'ml_m': {
                    'n_estimators': [100, 300],
                    'max_depth': [6, 10],
                    'min_samples_leaf': [5, 10],
                    'max_features': ['sqrt', 10]
                }
            },
            'decision_tree': {
                'ml_g': {
                    'max_depth': [6, 10],
                    'min_samples_split': [20, 50],
                    'min_samples_leaf': [10, 20],
                    'ccp_alpha': [0.0, 0.01]
                },
                'ml_m': {
                    'max_depth': [6, 10],
                    'min_samples_split': [20, 50],
                    'min_samples_leaf': [10, 20],
                    'ccp_alpha': [0.0, 0.01]
                }
            },
            'xgboost': {
                'ml_g': {
                    'n_estimators': [100, 300],
                    'max_depth': [3, 6],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                },
                'ml_m': {
                    'n_estimators': [100, 300],
                    'max_depth': [3, 6],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            }
        }
    
    def estimate_treatment_effects(
        self, 
        dml_data: DoubleMLData, 
        methods: Optional[List[str]] = None,
        tune_hyperparameters: bool = True
    ) -> Dict[str, CausalEstimate]:
        """
        Estimate treatment effects using multiple methods.
        
        Args:
            dml_data: Prepared DoubleML data
            methods: List of methods to use. If None, uses all available methods
            tune_hyperparameters: Whether to tune hyperparameters
            
        Returns:
            Dictionary mapping method names to causal estimates
        """
        if methods is None:
            methods = ['logistic', 'random_forest', 'decision_tree', 'xgboost']
        
        learners = self._get_base_learners()
        param_grids = self._get_hyperparameter_grids()
        
        estimates = {}
        
        for method in methods:
            logger.info(f"Estimating treatment effects using {method}")
            
            # Create Double ML IRM model
            dml_model = dml.DoubleMLIRM(
                dml_data,
                ml_g=learners[method]['ml_g'],
                ml_m=learners[method]['ml_m'],
                n_folds=self.n_folds
            )
            
            # Tune hyperparameters if requested and available
            if tune_hyperparameters and method in param_grids:
                logger.info(f"Tuning hyperparameters for {method}")
                dml_model.tune(param_grids[method], search_mode='grid_search')
            
            # Fit the model
            dml_model.fit(store_predictions=True)
            
            # Store model
            self.models[method] = dml_model
            
            # Extract results
            summary = dml_model.summary
            
            estimates[method] = CausalEstimate(
                coefficient=summary.loc[0, 'coef'],
                std_error=summary.loc[0, 'std err'],
                ci_lower=summary.loc[0, '2.5 %'],
                ci_upper=summary.loc[0, '97.5 %'],
                p_value=summary.loc[0, 'P>|t|'],
                method=method
            )
            
            logger.info(f"{method} - Coefficient: {estimates[method].coefficient:.6f}, "
                       f"P-value: {estimates[method].p_value:.6f}")
        
        self.results = estimates
        return estimates
    
    def evaluate_learner_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the performance of nuisance learners.
        
        Returns:
            Dictionary with performance metrics for each method
        """
        def logloss_metric(y_true, y_pred):
            """Safe log loss calculation."""
            subset = np.logical_not(np.isnan(y_true))
            if np.sum(subset) == 0:
                return np.nan
            return log_loss(y_true[subset], y_pred[subset])
        
        performance = {}
        
        for method, model in self.models.items():
            try:
                metrics = model.evaluate_learners(metric=logloss_metric)
                performance[method] = {
                    'ml_g0_logloss': float(metrics['ml_g0'][0][0]),
                    'ml_g1_logloss': float(metrics['ml_g1'][0][0]),
                    'ml_m_logloss': float(metrics['ml_m'][0][0])
                }
            except Exception as e:
                logger.warning(f"Could not evaluate learners for {method}: {e}")
                performance[method] = {'error': str(e)}
        
        return performance
    
    def run_placebo_test(
        self, 
        df: pd.DataFrame, 
        placebo_treatment: str = 'gender'
    ) -> CausalEstimate:
        """
        Run placebo test using a variable that should have no causal effect.
        
        Args:
            df: Preprocessed dataset
            placebo_treatment: Column to use as placebo treatment
            
        Returns:
            Causal estimate for placebo treatment
        """
        logger.info(f"Running placebo test with {placebo_treatment} as treatment")
        
        # Prepare data with placebo treatment
        placebo_data = self.prepare_data(
            df, 
            treatment_col=placebo_treatment,
            exclude_cols=['A1C_tested', 'A1Cresult_ord']
        )
        
        # Use simple logistic regression for placebo test
        dml_model = dml.DoubleMLIRM(
            placebo_data,
            ml_g=LogisticRegression(penalty=None, solver='newton-cg', max_iter=1000),
            ml_m=LogisticRegression(penalty=None, solver='newton-cg', max_iter=1000),
            n_folds=self.n_folds
        )
        
        dml_model.fit()
        summary = dml_model.summary
        
        return CausalEstimate(
            coefficient=summary.loc[0, 'coef'],
            std_error=summary.loc[0, 'std err'],
            ci_lower=summary.loc[0, '2.5 %'],
            ci_upper=summary.loc[0, '97.5 %'],
            p_value=summary.loc[0, 'P>|t|'],
            method='placebo_test'
        )
    
    def analyze_heterogeneous_effects(
        self, 
        df: pd.DataFrame, 
        subgroup_vars: List[str],
        method: str = 'random_forest'
    ) -> Dict[str, Dict[str, CausalEstimate]]:
        """
        Analyze heterogeneous treatment effects across subgroups.
        
        Args:
            df: Preprocessed dataset
            subgroup_vars: Variables to define subgroups
            method: Method to use for estimation
            
        Returns:
            Dictionary mapping subgroup variables to their effect estimates
        """
        heterogeneous_results = {}
        
        for var in subgroup_vars:
            logger.info(f"Analyzing heterogeneous effects by {var}")
            subgroup_results = {}
            
            # Get unique values for subgrouping
            unique_values = df[var].unique()
            
            for value in unique_values:
                if pd.isna(value):
                    continue
                    
                # Create subgroup
                subgroup_df = df[df[var] == value].copy()
                
                if len(subgroup_df) < 100:  # Skip small subgroups
                    logger.warning(f"Skipping subgroup {var}={value} (n={len(subgroup_df)})")
                    continue
                
                try:
                    # Prepare data for subgroup
                    subgroup_data = self.prepare_data(subgroup_df)
                    
                    # Estimate effects
                    estimates = self.estimate_treatment_effects(
                        subgroup_data, 
                        methods=[method],
                        tune_hyperparameters=False
                    )
                    
                    subgroup_results[str(value)] = estimates[method]
                    
                except Exception as e:
                    logger.error(f"Error analyzing subgroup {var}={value}: {e}")
            
            heterogeneous_results[var] = subgroup_results
        
        return heterogeneous_results