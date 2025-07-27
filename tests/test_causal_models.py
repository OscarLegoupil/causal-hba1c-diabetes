"""
Unit tests for causal inference models.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from causal_hba1c.models.causal_models import CausalInferenceEngine, CausalEstimate
from doubleml import DoubleMLData


class TestCausalEstimate(unittest.TestCase):
    """Test cases for CausalEstimate dataclass."""
    
    def test_initialization(self):
        """Test CausalEstimate initialization."""
        estimate = CausalEstimate(
            coefficient=-0.005,
            std_error=0.002,
            ci_lower=-0.009,
            ci_upper=-0.001,
            p_value=0.02,
            method='random_forest'
        )
        
        self.assertEqual(estimate.coefficient, -0.005)
        self.assertEqual(estimate.std_error, 0.002)
        self.assertEqual(estimate.method, 'random_forest')
    
    def test_is_significant(self):
        """Test significance testing."""
        # Significant result
        sig_estimate = CausalEstimate(
            coefficient=-0.005, std_error=0.002, ci_lower=-0.009,
            ci_upper=-0.001, p_value=0.02, method='test'
        )
        self.assertTrue(sig_estimate.is_significant)
        
        # Non-significant result
        nonsig_estimate = CausalEstimate(
            coefficient=-0.001, std_error=0.002, ci_lower=-0.005,
            ci_upper=0.003, p_value=0.6, method='test'
        )
        self.assertFalse(nonsig_estimate.is_significant)


class TestCausalInferenceEngine(unittest.TestCase):
    """Test cases for CausalInferenceEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = CausalInferenceEngine(n_folds=3, random_state=42)
        
        # Create sample data
        np.random.seed(42)
        n_samples = 200
        
        self.sample_data = pd.DataFrame({
            'A1C_tested': np.random.binomial(1, 0.3, n_samples),
            'readmitted': np.random.binomial(1, 0.1, n_samples),
            'age': np.random.randint(0, 10, n_samples),
            'gender': np.random.binomial(1, 0.5, n_samples),
            'time_in_hospital': np.random.randint(1, 10, n_samples),
            'num_medications': np.random.randint(1, 20, n_samples),
            'diabetesMed': np.random.binomial(1, 0.7, n_samples),
            'emergency': np.random.binomial(1, 0.4, n_samples),
            'diag_Diabetes': np.random.randint(0, 4, n_samples),
        })
    
    def test_initialization(self):
        """Test engine initialization."""
        self.assertEqual(self.engine.n_folds, 3)
        self.assertEqual(self.engine.random_state, 42)
        self.assertIsInstance(self.engine.models, dict)
        self.assertIsInstance(self.engine.results, dict)
    
    def test_prepare_data(self):
        """Test data preparation for DoubleML."""
        dml_data = self.engine.prepare_data(self.sample_data)
        
        self.assertIsInstance(dml_data, DoubleMLData)
        self.assertEqual(dml_data.y_col, 'readmitted')
        self.assertEqual(dml_data.d_cols, ['A1C_tested'])
        
        # Should exclude treatment, outcome, and A1Cresult_ord from covariates
        expected_covariates = [
            'age', 'gender', 'time_in_hospital', 'num_medications',
            'diabetesMed', 'emergency', 'diag_Diabetes'
        ]
        for covariate in expected_covariates:
            self.assertIn(covariate, dml_data.x_cols)
        
        self.assertNotIn('A1C_tested', dml_data.x_cols)
        self.assertNotIn('readmitted', dml_data.x_cols)
    
    def test_get_base_learners(self):
        """Test base learner configuration."""
        learners = self.engine._get_base_learners()
        
        expected_methods = ['logistic', 'random_forest', 'decision_tree', 'xgboost']
        for method in expected_methods:
            self.assertIn(method, learners)
            self.assertIn('ml_g', learners[method])
            self.assertIn('ml_m', learners[method])
    
    def test_get_hyperparameter_grids(self):
        """Test hyperparameter grid configuration."""
        param_grids = self.engine._get_hyperparameter_grids()
        
        # Should have grids for tunable methods
        tunable_methods = ['random_forest', 'decision_tree', 'xgboost']
        for method in tunable_methods:
            self.assertIn(method, param_grids)
            self.assertIn('ml_g', param_grids[method])
            self.assertIn('ml_m', param_grids[method])
    
    @patch('doubleml.DoubleMLIRM')
    def test_estimate_treatment_effects(self, mock_dml_irm):
        """Test treatment effect estimation."""
        # Mock DoubleML model
        mock_model = Mock()
        mock_summary = pd.DataFrame({
            'coef': [-0.005],
            'std err': [0.002],
            '2.5 %': [-0.009],
            '97.5 %': [-0.001],
            'P>|t|': [0.02]
        })
        mock_model.fit.return_value = mock_model
        mock_model.summary = mock_summary
        mock_dml_irm.return_value = mock_model
        
        # Prepare data
        dml_data = self.engine.prepare_data(self.sample_data)
        
        # Test estimation
        estimates = self.engine.estimate_treatment_effects(
            dml_data, 
            methods=['logistic'],
            tune_hyperparameters=False
        )
        
        # Assertions
        self.assertIn('logistic', estimates)
        self.assertIsInstance(estimates['logistic'], CausalEstimate)
        self.assertEqual(estimates['logistic'].coefficient, -0.005)
        self.assertEqual(estimates['logistic'].method, 'logistic')
    
    def test_run_placebo_test(self):
        """Test placebo test functionality."""
        # Add required columns for placebo test
        self.sample_data['A1Cresult_ord'] = np.random.randint(-1, 3, len(self.sample_data))
        
        with patch('doubleml.DoubleMLIRM') as mock_dml_irm:
            # Mock model
            mock_model = Mock()
            mock_summary = pd.DataFrame({
                'coef': [0.001],
                'std err': [0.003],
                '2.5 %': [-0.005],
                '97.5 %': [0.007],
                'P>|t|': [0.7]
            })
            mock_model.fit.return_value = mock_model
            mock_model.summary = mock_summary
            mock_dml_irm.return_value = mock_model
            
            # Run placebo test
            placebo_result = self.engine.run_placebo_test(
                self.sample_data, 
                placebo_treatment='gender'
            )
            
            # Assertions
            self.assertIsInstance(placebo_result, CausalEstimate)
            self.assertEqual(placebo_result.method, 'placebo_test')
            self.assertFalse(placebo_result.is_significant)  # Should not be significant
    
    def test_analyze_heterogeneous_effects(self):
        """Test heterogeneous effects analysis."""
        # Add age groups
        self.sample_data['age_group'] = np.random.choice(
            ['young', 'middle', 'old'], len(self.sample_data)
        )
        
        with patch.object(self.engine, 'estimate_treatment_effects') as mock_estimate:
            # Mock estimates for subgroups
            mock_estimate.return_value = {
                'random_forest': CausalEstimate(
                    coefficient=-0.01, std_error=0.005, ci_lower=-0.02,
                    ci_upper=0.0, p_value=0.05, method='random_forest'
                )
            }
            
            # Test heterogeneous analysis
            het_results = self.engine.analyze_heterogeneous_effects(
                self.sample_data,
                subgroup_vars=['age_group'],
                method='random_forest'
            )
            
            # Assertions
            self.assertIn('age_group', het_results)
            self.assertIsInstance(het_results['age_group'], dict)
            
            # Should have results for each age group
            age_groups = self.sample_data['age_group'].unique()
            for group in age_groups:
                if str(group) in het_results['age_group']:
                    self.assertIsInstance(
                        het_results['age_group'][str(group)], 
                        CausalEstimate
                    )


class TestCausalModelIntegration(unittest.TestCase):
    """Integration tests for causal models."""
    
    def setUp(self):
        """Set up integration test data."""
        np.random.seed(123)
        n_samples = 500
        
        # Create synthetic data with some realistic relationships
        age = np.random.randint(0, 10, n_samples)
        diabetes_severity = np.random.normal(0, 1, n_samples)
        
        # A1C testing more likely for older patients and those with diabetes
        a1c_prob = 0.2 + 0.05 * age + 0.1 * (diabetes_severity > 0)
        a1c_tested = np.random.binomial(1, np.clip(a1c_prob, 0, 1), n_samples)
        
        # Readmission influenced by age, diabetes severity, and A1C testing
        readmit_prob = 0.05 + 0.01 * age + 0.05 * diabetes_severity - 0.02 * a1c_tested
        readmitted = np.random.binomial(1, np.clip(readmit_prob, 0, 1), n_samples)
        
        self.synthetic_data = pd.DataFrame({
            'A1C_tested': a1c_tested,
            'readmitted': readmitted,
            'age': age,
            'diabetes_severity': diabetes_severity,
            'gender': np.random.binomial(1, 0.5, n_samples),
            'time_in_hospital': np.random.randint(1, 10, n_samples),
            'num_medications': np.random.randint(1, 20, n_samples),
            'emergency': np.random.binomial(1, 0.3, n_samples),
        })
    
    def test_full_causal_analysis_pipeline(self):
        """Test the complete causal analysis pipeline."""
        engine = CausalInferenceEngine(n_folds=3, random_state=42)
        
        # Prepare data
        dml_data = engine.prepare_data(self.synthetic_data)
        self.assertIsInstance(dml_data, DoubleMLData)
        
        # We'll mock the actual ML fitting since it's computationally expensive
        with patch('doubleml.DoubleMLIRM') as mock_dml_irm:
            # Mock successful model fitting
            mock_model = Mock()
            mock_summary = pd.DataFrame({
                'coef': [-0.02],  # True effect we built in
                'std err': [0.01],
                '2.5 %': [-0.04],
                '97.5 %': [0.00],
                'P>|t|': [0.05]
            })
            mock_model.fit.return_value = mock_model
            mock_model.summary = mock_summary
            mock_model.evaluate_learners.return_value = {
                'ml_g0': np.array([[0.3]]),
                'ml_g1': np.array([[0.32]]),
                'ml_m': np.array([[0.4]])
            }
            mock_dml_irm.return_value = mock_model
            
            # Estimate treatment effects
            estimates = engine.estimate_treatment_effects(
                dml_data,
                methods=['logistic', 'random_forest'],
                tune_hyperparameters=False
            )
            
            # Assertions
            self.assertEqual(len(estimates), 2)
            self.assertIn('logistic', estimates)
            self.assertIn('random_forest', estimates)
            
            for method, estimate in estimates.items():
                self.assertIsInstance(estimate, CausalEstimate)
                self.assertEqual(estimate.coefficient, -0.02)
                self.assertEqual(estimate.method, method)
            
            # Test model performance evaluation
            performance = engine.evaluate_learner_performance()
            self.assertIn('logistic', performance)
            self.assertIn('random_forest', performance)


if __name__ == '__main__':
    unittest.main()