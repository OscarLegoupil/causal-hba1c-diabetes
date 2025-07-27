"""
Utility functions for the causal inference analysis.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import pickle


logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
    """
    log_level = getattr(logging, level.upper())
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Save analysis results to file.
    
    Args:
        results: Dictionary containing analysis results
        filepath: Path to save results
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        # Convert non-serializable objects to strings
        serializable_results = {}
        for key, value in results.items():
            try:
                json.dumps(value)
                serializable_results[key] = value
            except (TypeError, ValueError):
                serializable_results[key] = str(value)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    elif filepath.suffix == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    logger.info(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load analysis results from file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Dictionary containing analysis results
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            results = json.load(f)
    
    elif filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
    
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    logger.info(f"Results loaded from {filepath}")
    return results


def calculate_summary_statistics(df: pd.DataFrame, group_col: str = None) -> pd.DataFrame:
    """
    Calculate summary statistics for key variables.
    
    Args:
        df: Dataset
        group_col: Optional column to group by
        
    Returns:
        DataFrame with summary statistics
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if group_col and group_col in df.columns:
        summary = df.groupby(group_col)[numeric_cols].agg(['count', 'mean', 'std', 'min', 'max'])
    else:
        summary = df[numeric_cols].agg(['count', 'mean', 'std', 'min', 'max']).T
    
    return summary


def check_balance(df: pd.DataFrame, treatment_col: str, covariates: List[str]) -> pd.DataFrame:
    """
    Check covariate balance between treatment and control groups.
    
    Args:
        df: Dataset
        treatment_col: Name of treatment variable
        covariates: List of covariate columns
        
    Returns:
        DataFrame with balance statistics
    """
    balance_stats = []
    
    for covariate in covariates:
        if covariate not in df.columns:
            continue
            
        treated = df[df[treatment_col] == 1][covariate]
        control = df[df[treatment_col] == 0][covariate]
        
        # Calculate standardized mean difference
        if treated.std() + control.std() > 0:
            smd = (treated.mean() - control.mean()) / np.sqrt((treated.var() + control.var()) / 2)
        else:
            smd = 0
        
        balance_stats.append({
            'covariate': covariate,
            'treated_mean': treated.mean(),
            'control_mean': control.mean(),
            'treated_std': treated.std(),
            'control_std': control.std(),
            'standardized_mean_diff': smd
        })
    
    return pd.DataFrame(balance_stats)


def create_age_groups(df: pd.DataFrame, age_col: str = 'age') -> pd.DataFrame:
    """
    Create age group categories for heterogeneous analysis.
    
    Args:
        df: Dataset
        age_col: Name of age column
        
    Returns:
        DataFrame with age groups added
    """
    df = df.copy()
    
    # Map ordinal age to age groups
    age_group_mapping = {
        0: 'pediatric', 1: 'pediatric',  # [0-10), [10-20)
        2: 'young_adult', 3: 'young_adult',  # [20-30), [30-40)
        4: 'middle_aged', 5: 'middle_aged', 6: 'middle_aged',  # [40-50), [50-60), [60-70)
        7: 'elderly', 8: 'elderly', 9: 'elderly'  # [70-80), [80-90), [90-100)
    }
    
    df['age_group'] = df[age_col].map(age_group_mapping)
    
    return df


def calculate_treatment_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate treatment intensity based on multiple factors.
    
    Args:
        df: Dataset
        
    Returns:
        DataFrame with treatment intensity measures
    """
    df = df.copy()
    
    # Create treatment intensity score
    intensity_components = [
        'A1C_tested',
        'max_glu_serum_tested',
        'change',
        'diabetesMed'
    ]
    
    available_components = [col for col in intensity_components if col in df.columns]
    
    if available_components:
        df['treatment_intensity'] = df[available_components].sum(axis=1)
    else:
        df['treatment_intensity'] = 0
    
    return df


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and return quality metrics.
    
    Args:
        df: Dataset to validate
        
    Returns:
        Dictionary with data quality metrics
    """
    quality_metrics = {}
    
    # Basic metrics
    quality_metrics['n_observations'] = len(df)
    quality_metrics['n_features'] = len(df.columns)
    
    # Missing data
    missing_counts = df.isnull().sum()
    quality_metrics['missing_data'] = {
        'total_missing': missing_counts.sum(),
        'features_with_missing': (missing_counts > 0).sum(),
        'max_missing_feature': missing_counts.idxmax() if missing_counts.sum() > 0 else None,
        'max_missing_count': missing_counts.max()
    }
    
    # Data types
    quality_metrics['data_types'] = df.dtypes.value_counts().to_dict()
    
    # Duplicates
    quality_metrics['duplicates'] = df.duplicated().sum()
    
    # Outliers (for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_counts = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_counts[col] = outliers
    
    quality_metrics['outliers'] = outlier_counts
    
    # Treatment variable checks
    if 'A1C_tested' in df.columns:
        treatment_dist = df['A1C_tested'].value_counts(normalize=True)
        quality_metrics['treatment_distribution'] = treatment_dist.to_dict()
    
    # Outcome variable checks
    if 'readmitted' in df.columns:
        outcome_dist = df['readmitted'].value_counts(normalize=True)
        quality_metrics['outcome_distribution'] = outcome_dist.to_dict()
    
    return quality_metrics


def format_results_table(estimates: Dict[str, Any], title: str = "Treatment Effect Estimates") -> str:
    """
    Format results as a nice table for reporting.
    
    Args:
        estimates: Dictionary of causal estimates
        title: Title for the table
        
    Returns:
        Formatted table string
    """
    table_lines = [f"\n{title}", "=" * len(title)]
    
    headers = ["Method", "Coefficient", "Std Error", "95% CI", "P-value", "Significant"]
    table_lines.append(" | ".join(f"{h:>12}" for h in headers))
    table_lines.append("-" * (13 * len(headers) + len(headers) - 1))
    
    for method, est in estimates.items():
        if hasattr(est, 'coefficient'):
            significance = "Yes" if est.is_significant else "No"
            ci_str = f"[{est.ci_lower:.4f}, {est.ci_upper:.4f}]"
            
            row = [
                method[:12],
                f"{est.coefficient:.6f}",
                f"{est.std_error:.6f}",
                ci_str,
                f"{est.p_value:.4f}",
                significance
            ]
            table_lines.append(" | ".join(f"{cell:>12}" for cell in row))
    
    return "\n".join(table_lines)


def ensure_directory(path: str) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj