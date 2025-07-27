"""
Unit tests for data loading and preprocessing modules.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from causal_hba1c.data.loader import DiabetesDataLoader
from causal_hba1c.data.preprocessor import DiabetesPreprocessor


class TestDiabetesDataLoader(unittest.TestCase):
    """Test cases for DiabetesDataLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = DiabetesDataLoader()
    
    def test_initialization(self):
        """Test loader initialization."""
        self.assertEqual(self.loader.dataset_id, 296)
        self.assertIsNone(self.loader._raw_data)
        self.assertIsNone(self.loader._metadata)
    
    @patch('causal_hba1c.data.loader.fetch_ucirepo')
    def test_load_data_basic(self, mock_fetch):
        """Test basic data loading functionality."""
        # Mock data
        mock_features = pd.DataFrame({
            'age': ['[30-40)', '[50-60)'],
            'gender': ['Male', 'Female'],
            'A1Cresult': ['None', '>7']
        })
        mock_targets = pd.DataFrame({'readmitted': ['NO', '<30']})
        mock_ids = pd.DataFrame({'patient_nbr': [1, 2], 'encounter_id': [10, 20]})
        
        mock_data = Mock()
        mock_data.data.features = mock_features
        mock_data.data.targets = mock_targets
        mock_data.data.ids = mock_ids
        mock_data.metadata = {'name': 'test_dataset'}
        
        mock_fetch.return_value = mock_data
        
        # Test loading
        result = self.loader.load_data(remove_duplicates=False)
        
        # Assertions
        self.assertEqual(len(result), 2)
        self.assertIn('age', result.columns)
        self.assertIn('readmitted', result.columns)
        self.assertIn('patient_nbr', result.columns)
        mock_fetch.assert_called_once_with(id=296)
    
    @patch('causal_hba1c.data.loader.fetch_ucirepo')
    def test_load_data_with_duplicates(self, mock_fetch):
        """Test data loading with duplicate removal."""
        # Mock data with duplicates
        mock_features = pd.DataFrame({
            'age': ['[30-40)', '[50-60)', '[30-40)'],
            'gender': ['Male', 'Female', 'Male']
        })
        mock_targets = pd.DataFrame({'readmitted': ['NO', '<30', 'NO']})
        mock_ids = pd.DataFrame({'patient_nbr': [1, 2, 1], 'encounter_id': [10, 20, 30]})
        
        mock_data = Mock()
        mock_data.data.features = mock_features
        mock_data.data.targets = mock_targets
        mock_data.data.ids = mock_ids
        mock_data.metadata = {}
        
        mock_fetch.return_value = mock_data
        
        # Test loading with duplicate removal
        result = self.loader.load_data(remove_duplicates=True)
        
        # Should have 2 unique patients
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result['patient_nbr'].unique()), 2)


class TestDiabetesPreprocessor(unittest.TestCase):
    """Test cases for DiabetesPreprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DiabetesPreprocessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'encounter_id': [1, 2, 3],
            'patient_nbr': [101, 102, 103],
            'age': ['[30-40)', '[50-60)', '[70-80)'],
            'gender': ['Male', 'Female', 'Male'],
            'race': ['Caucasian', 'AfricanAmerican', 'Hispanic'],
            'admission_type_id': [1, 3, 2],
            'discharge_disposition_id': [1, 2, 6],
            'time_in_hospital': [3, 5, 2],
            'num_medications': [5, 8, 3],
            'diag_1': ['250', '401', '780'],
            'diag_2': ['401', '250', '250'],
            'diag_3': [np.nan, '780', '401'],
            'A1Cresult': ['>7', 'None', 'Norm'],
            'max_glu_serum': ['None', '>200', 'None'],
            'change': ['Ch', 'No', 'Ch'],
            'diabetesMed': ['Yes', 'No', 'Yes'],
            'readmitted': ['<30', 'NO', '>30'],
            'weight': [np.nan, np.nan, np.nan],
            'payer_code': [np.nan, 'BC', np.nan],
            'medical_specialty': ['InternalMedicine', np.nan, 'Cardiology']
        })
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        self.assertIsInstance(self.preprocessor.age_mapping, dict)
        self.assertIsInstance(self.preprocessor.diagnosis_groups, dict)
        self.assertIn('Diabetes', self.preprocessor.diagnosis_groups)
        self.assertEqual(self.preprocessor.diagnosis_groups['Diabetes'], [250])
    
    def test_map_icd9_to_group(self):
        """Test ICD-9 code mapping."""
        # Test diabetes mapping
        self.assertEqual(self.preprocessor._map_icd9_to_group('250'), 'Diabetes')
        self.assertEqual(self.preprocessor._map_icd9_to_group('250.0'), 'Diabetes')
        
        # Test circulatory mapping
        self.assertEqual(self.preprocessor._map_icd9_to_group('401'), 'Circulatory')
        
        # Test unknown mapping
        self.assertEqual(self.preprocessor._map_icd9_to_group('999'), 'Unknown')
        self.assertEqual(self.preprocessor._map_icd9_to_group(np.nan), 'Unknown')
        self.assertEqual(self.preprocessor._map_icd9_to_group('V70'), 'External causes')
    
    def test_preprocess_basic_transformations(self):
        """Test basic preprocessing transformations."""
        result = self.preprocessor.preprocess(self.sample_data)
        
        # Check treatment variables created
        self.assertIn('A1C_tested', result.columns)
        self.assertIn('max_glu_serum_tested', result.columns)
        
        # Check age transformation
        self.assertTrue(all(isinstance(x, (int, np.integer)) for x in result['age']))
        self.assertEqual(result['age'].iloc[0], 3)  # [30-40) -> 3
        
        # Check gender transformation
        self.assertEqual(result['gender'].iloc[0], 0)  # Male -> 0
        self.assertEqual(result['gender'].iloc[1], 1)  # Female -> 1
        
        # Check target variable
        self.assertEqual(result['readmitted'].iloc[0], 1)  # <30 -> 1
        self.assertEqual(result['readmitted'].iloc[1], 0)  # NO -> 0
        self.assertEqual(result['readmitted'].iloc[2], 0)  # >30 -> 0
    
    def test_preprocess_removes_columns(self):
        """Test that unnecessary columns are removed."""
        result = self.preprocessor.preprocess(self.sample_data)
        
        # Should not contain these columns
        removed_cols = ['encounter_id', 'patient_nbr', 'weight', 'payer_code']
        for col in removed_cols:
            self.assertNotIn(col, result.columns)
    
    def test_preprocess_creates_dummies(self):
        """Test that dummy variables are created correctly."""
        result = self.preprocessor.preprocess(self.sample_data)
        
        # Should have race dummies
        race_dummies = [col for col in result.columns if col.startswith('race_')]
        self.assertGreater(len(race_dummies), 0)
        
        # Should have diagnosis variables
        diag_vars = [col for col in result.columns if col.startswith('diag_')]
        self.assertGreater(len(diag_vars), 0)
    
    def test_get_feature_groups(self):
        """Test feature grouping functionality."""
        processed_data = self.preprocessor.preprocess(self.sample_data)
        feature_groups = self.preprocessor.get_feature_groups(processed_data)
        
        self.assertIsInstance(feature_groups, dict)
        expected_groups = ['demographics', 'medical_history', 'test_results', 
                          'admission_context', 'diagnoses', 'treatment_vars']
        for group in expected_groups:
            self.assertIn(group, feature_groups)
            self.assertIsInstance(feature_groups[group], list)


class TestDataIntegration(unittest.TestCase):
    """Integration tests for data loading and preprocessing."""
    
    @patch('causal_hba1c.data.loader.fetch_ucirepo')
    def test_full_pipeline(self, mock_fetch):
        """Test the complete data loading and preprocessing pipeline."""
        # Create more comprehensive mock data
        n_samples = 100
        mock_features = pd.DataFrame({
            'age': np.random.choice(['[30-40)', '[50-60)', '[70-80)'], n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'race': np.random.choice(['Caucasian', 'AfricanAmerican'], n_samples),
            'admission_type_id': np.random.choice([1, 2, 3], n_samples),
            'discharge_disposition_id': np.random.choice([1, 2, 6], n_samples),
            'time_in_hospital': np.random.randint(1, 10, n_samples),
            'num_medications': np.random.randint(1, 20, n_samples),
            'diag_1': np.random.choice(['250', '401', '780'], n_samples),
            'A1Cresult': np.random.choice(['>7', 'None', 'Norm'], n_samples),
            'change': np.random.choice(['Ch', 'No'], n_samples),
            'diabetesMed': np.random.choice(['Yes', 'No'], n_samples),
        })
        
        mock_targets = pd.DataFrame({
            'readmitted': np.random.choice(['<30', 'NO', '>30'], n_samples)
        })
        
        mock_ids = pd.DataFrame({
            'patient_nbr': range(n_samples),
            'encounter_id': range(100, 100 + n_samples)
        })
        
        mock_data = Mock()
        mock_data.data.features = mock_features
        mock_data.data.targets = mock_targets
        mock_data.data.ids = mock_ids
        mock_data.metadata = {}
        
        mock_fetch.return_value = mock_data
        
        # Run full pipeline
        loader = DiabetesDataLoader()
        raw_data = loader.load_data()
        
        preprocessor = DiabetesPreprocessor()
        processed_data = preprocessor.preprocess(raw_data)
        
        # Assertions
        self.assertEqual(len(processed_data), n_samples)
        self.assertIn('A1C_tested', processed_data.columns)
        self.assertIn('readmitted', processed_data.columns)
        
        # Check data types
        self.assertTrue(processed_data['A1C_tested'].dtype in [np.int64, np.int32, int])
        self.assertTrue(processed_data['readmitted'].dtype in [np.int64, np.int32, int])
        
        # Check no missing values in key columns
        self.assertEqual(processed_data['A1C_tested'].isnull().sum(), 0)
        self.assertEqual(processed_data['readmitted'].isnull().sum(), 0)


if __name__ == '__main__':
    unittest.main()