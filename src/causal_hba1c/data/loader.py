"""
Data loading module for UCI Diabetes dataset.
"""

import pandas as pd
from ucimlrepo import fetch_ucirepo
from typing import Tuple, Optional
import logging


logger = logging.getLogger(__name__)


class DiabetesDataLoader:
    """Loads and provides access to the UCI Diabetes 130-US hospitals dataset."""
    
    def __init__(self, dataset_id: int = 296):
        """
        Initialize the data loader.
        
        Args:
            dataset_id: UCI ML Repository dataset ID (default: 296 for diabetes dataset)
        """
        self.dataset_id = dataset_id
        self._raw_data = None
        self._metadata = None
        
    def load_data(self, remove_duplicates: bool = True) -> pd.DataFrame:
        """
        Load the diabetes dataset from UCI ML Repository.
        
        Args:
            remove_duplicates: Whether to remove duplicate patient entries
            
        Returns:
            Combined dataset with features, targets, and IDs
        """
        logger.info(f"Loading diabetes dataset (ID: {self.dataset_id})")
        
        # Fetch dataset from UCI repository
        diabetes_data = fetch_ucirepo(id=self.dataset_id)
        
        # Store metadata
        self._metadata = diabetes_data.metadata
        
        # Combine features, targets, and IDs
        X = diabetes_data.data.features
        y = diabetes_data.data.targets
        ids = diabetes_data.data.ids
        
        df = pd.concat([ids, X, y], axis=1)
        
        if remove_duplicates:
            initial_size = len(df)
            df = df.drop_duplicates(subset="patient_nbr", keep="first")
            logger.info(f"Removed {initial_size - len(df)} duplicate patient entries")
            
        self._raw_data = df
        logger.info(f"Loaded dataset with {len(df)} observations and {len(df.columns)} features")
        
        return df
    
    def get_metadata(self) -> Optional[dict]:
        """Get dataset metadata."""
        return self._metadata
    
    def get_variable_info(self) -> Optional[dict]:
        """Get variable information from the dataset."""
        if self._metadata:
            return self._metadata.get('variables', None)
        return None
    
    def describe_dataset(self) -> None:
        """Print dataset description and basic statistics."""
        if self._raw_data is None:
            logger.error("No data loaded. Call load_data() first.")
            return
            
        print("Dataset Overview:")
        print("=" * 50)
        print(f"Shape: {self._raw_data.shape}")
        print(f"Unique patients: {self._raw_data['patient_nbr'].nunique()}")
        print(f"Total encounters: {len(self._raw_data)}")
        
        print("\nTarget variable distribution:")
        target_dist = self._raw_data['readmitted'].value_counts(normalize=True)
        for category, proportion in target_dist.items():
            print(f"  {category}: {proportion:.3f}")
        
        print("\nMissing data summary:")
        missing_summary = self._raw_data.isnull().sum()
        missing_pct = (missing_summary / len(self._raw_data)) * 100
        
        for col in self._raw_data.columns:
            if missing_pct[col] > 0:
                print(f"  {col}: {missing_pct[col]:.1f}%")