"""
Data preprocessing module for causal inference analysis.
"""

import pandas as pd
import numpy as np
from math import floor
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)


class DiabetesPreprocessor:
    """Preprocesses diabetes dataset for causal inference analysis."""
    
    def __init__(self):
        """Initialize the preprocessor with ICD-9 diagnosis mappings."""
        self.age_mapping = {
            '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3,
            '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7,
            '[80-90)': 8, '[90-100)': 9
        }
        
        # ICD-9 diagnosis groupings
        self.diagnosis_groups = self._create_diagnosis_groups()
        
        # Columns to drop (high missingness or irrelevant)
        self.columns_to_drop = [
            "encounter_id", "patient_nbr", "payer_code", "weight", 
            "medical_specialty", "admission_source_id",
            # Medication columns (mostly unchanged)
            "metformin", "glimepiride", "glipizide", "glyburide", 
            "pioglitazone", "rosiglitazone", "insulin", "repaglinide",
            "nateglinide", "chlorpropamide", "acetohexamide", "tolbutamide",
            "acarbose", "miglitol", "troglitazone", "tolazamide", 
            "examide", "citoglipton", "glyburide-metformin",
            "glipizide-metformin", "glimepiride-pioglitazone",
            "metformin-rosiglitazone", "metformin-pioglitazone"
        ]
        
    def _create_diagnosis_groups(self) -> Dict[str, List[int]]:
        """Create ICD-9 diagnosis code groupings."""
        interval = lambda x, y: list(range(x, y + 1))
        
        return {
            "Circulatory": interval(390, 433) + interval(435, 459) + [785],
            "Respiratory": interval(460, 519) + [786],
            "Digestive": interval(520, 579) + [786],
            "Diabetes": [250],
            "Injury": interval(800, 999),
            "Musculoskeletal": interval(710, 739),
            "Genitourinary": interval(580, 629) + [788],
            "Neoplasms": interval(140, 239),
            "Endocrine": interval(240, 249) + interval(251, 279),
            "General symptoms": [780, 781, 784] + interval(790, 799),
            "Skin": interval(680, 709) + [782],
            "Infection": interval(1, 139),
            "Mental": interval(290, 319),
            "External causes": [1000],
            "Blood": interval(280, 289),
            "Nervous": interval(320, 359),
            "Pregnancy": interval(630, 679),
            "Sense organs": interval(360, 389),
            "Congenital": interval(740, 759),
            "Occlusion of cerebral arteries": [434],
            "Unknown": [-1]
        }
    
    def _map_icd9_to_group(self, code: str) -> str:
        """Map ICD-9 code to diagnostic group."""
        if pd.isna(code):
            return "Unknown"
            
        try:
            if str(code).startswith(('V', 'E')):
                code_num = 1000
            else:
                code_num = floor(float(code))
        except (ValueError, TypeError):
            return "Unknown"
        
        for group_name, codes in self.diagnosis_groups.items():
            if code_num in codes:
                return group_name
        
        return "Unknown"
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive preprocessing to the dataset.
        
        Args:
            df: Raw diabetes dataset
            
        Returns:
            Preprocessed dataset ready for causal analysis
        """
        logger.info("Starting data preprocessing")
        df_processed = df.copy()
        
        # Create treatment and additional test indicators
        df_processed['A1C_tested'] = (~df_processed['A1Cresult'].isna()).astype(int)
        df_processed['max_glu_serum_tested'] = (~df_processed['max_glu_serum'].isna()).astype(int)
        
        # Drop irrelevant columns
        df_processed = df_processed.drop(
            columns=[col for col in self.columns_to_drop if col in df_processed.columns]
        )
        
        # Process age groups to ordinal
        df_processed['age'] = df_processed['age'].map(self.age_mapping)
        
        # Process gender (1 = Female, 0 = Male/Unknown)
        df_processed['gender'] = (df_processed['gender'] == 'Female').astype(int)
        
        # Process admission type (1 = Emergency, 0 = Other)
        df_processed['emergency'] = df_processed['admission_type_id'].isin([1, 2]).astype(int)
        df_processed = df_processed.drop('admission_type_id', axis=1)
        
        # Process discharge disposition (1 = Home, 0 = Other)
        df_processed['sent_home'] = df_processed['discharge_disposition_id'].isin([1, 6, 8]).astype(int)
        df_processed = df_processed.drop('discharge_disposition_id', axis=1)
        
        # Process diagnosis codes
        for diag_col in ['diag_1', 'diag_2', 'diag_3']:
            if diag_col in df_processed.columns:
                df_processed[diag_col] = df_processed[diag_col].fillna("-1").apply(self._map_icd9_to_group)
        
        # Process test results to ordinal
        df_processed['A1Cresult_ord'] = df_processed['A1Cresult'].fillna("None").map({
            "None": -1, "Norm": 0, ">7": 1, ">8": 2
        })
        
        df_processed['max_glu_serum_ord'] = df_processed['max_glu_serum'].fillna("None").map({
            "None": -1, "Norm": 0, ">200": 1, ">300": 2
        })
        
        # Process binary variables
        df_processed['change'] = (df_processed['change'] == 'Ch').astype(int)
        df_processed['diabetesMed'] = (df_processed['diabetesMed'] == 'Yes').astype(int)
        
        # Process target variable (1 = readmitted <30 days, 0 = not readmitted <30 days)
        df_processed['readmitted'] = (df_processed['readmitted'] == '<30').astype(int)
        
        # Drop original columns that have been transformed
        cols_to_drop_final = ['A1Cresult', 'max_glu_serum']
        df_processed = df_processed.drop(
            columns=[col for col in cols_to_drop_final if col in df_processed.columns]
        )
        
        # Create dummy variables for categorical features
        categorical_cols = ['race'] + [col for col in df_processed.columns if col.startswith('diag_')]
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, dummy_na=False, drop_first=False)
        
        # Combine diagnosis variables with weights (primary=3, secondary=2, tertiary=1)
        diagnosis_groups = set()
        for col in df_processed.columns:
            if col.startswith('diag_1_'):
                diagnosis_groups.add(col.replace('diag_1_', ''))
        
        for disease in diagnosis_groups:
            diag_cols = [f"diag_{i}_{disease}" for i in range(1, 4)]
            existing_cols = [col for col in diag_cols if col in df_processed.columns]
            
            if len(existing_cols) >= 1:
                weights = [3, 2, 1][:len(existing_cols)]
                df_processed[f"diag_{disease}"] = sum(
                    df_processed[col] * weight for col, weight in zip(existing_cols, weights)
                )
                df_processed = df_processed.drop(columns=existing_cols)
        
        logger.info(f"Preprocessing complete. Final dataset shape: {df_processed.shape}")
        return df_processed
    
    def get_feature_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Categorize features into groups for analysis.
        
        Args:
            df: Preprocessed dataset
            
        Returns:
            Dictionary mapping feature group names to column lists
        """
        demographics = ['gender', 'age'] + [col for col in df.columns if col.startswith('race_')]
        
        medical_history = [
            'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
            'num_medications', 'number_outpatient', 'number_emergency', 
            'number_inpatient', 'number_diagnoses'
        ]
        
        test_results = ['A1Cresult_ord', 'max_glu_serum_ord', 'max_glu_serum_tested']
        
        admission_context = ['emergency', 'sent_home']
        
        diagnoses = [col for col in df.columns if col.startswith('diag_') and not col.endswith('_tested')]
        
        treatment_vars = ['change', 'diabetesMed']
        
        return {
            'demographics': demographics,
            'medical_history': medical_history,
            'test_results': test_results,
            'admission_context': admission_context,
            'diagnoses': diagnoses,
            'treatment_vars': treatment_vars
        }