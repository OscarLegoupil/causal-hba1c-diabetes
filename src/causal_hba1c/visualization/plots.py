"""
Visualization module for causal inference analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

from ..models.causal_models import CausalEstimate


logger = logging.getLogger(__name__)


class CausalVisualization:
    """Creates professional visualizations for causal inference analysis."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize visualization settings.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        plt.style.use('default')  # Use default style since seaborn styles may not be available
        sns.set_palette("husl")
        self.figsize = figsize
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'neutral': '#C73E1D',
            'light_gray': '#F5F5F5',
            'dark_gray': '#333333'
        }
    
    def plot_data_overview(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Create comprehensive data overview plots.
        
        Args:
            df: Dataset to visualize
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
        
        # Age distribution
        age_counts = df['age'].value_counts().sort_index()
        age_labels = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
                     '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
        axes[0, 0].bar(range(len(age_counts)), age_counts.values, color=self.colors['primary'])
        axes[0, 0].set_title('Age Distribution')
        axes[0, 0].set_xlabel('Age Group')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_xticks(range(len(age_labels)))
        axes[0, 0].set_xticklabels(age_labels, rotation=45)
        
        # Gender distribution
        gender_counts = df['gender'].value_counts()
        gender_labels = ['Male', 'Female']
        axes[0, 1].pie(gender_counts.values, labels=gender_labels, autopct='%1.1f%%',
                      colors=[self.colors['primary'], self.colors['secondary']])
        axes[0, 1].set_title('Gender Distribution')
        
        # Readmission rates
        readmit_counts = df['readmitted'].value_counts()
        readmit_labels = ['Not Readmitted', 'Readmitted <30 days']
        axes[0, 2].pie(readmit_counts.values, labels=readmit_labels, autopct='%1.1f%%',
                      colors=[self.colors['light_gray'], self.colors['neutral']])
        axes[0, 2].set_title('Readmission Distribution')
        
        # A1C testing rates
        a1c_counts = df['A1C_tested'].value_counts()
        a1c_labels = ['Not Tested', 'A1C Tested']
        axes[1, 0].pie(a1c_counts.values, labels=a1c_labels, autopct='%1.1f%%',
                      colors=[self.colors['light_gray'], self.colors['accent']])
        axes[1, 0].set_title('A1C Testing Distribution')
        
        # Time in hospital
        axes[1, 1].hist(df['time_in_hospital'], bins=20, color=self.colors['primary'], alpha=0.7)
        axes[1, 1].set_title('Time in Hospital Distribution')
        axes[1, 1].set_xlabel('Days')
        axes[1, 1].set_ylabel('Frequency')
        
        # Number of medications
        axes[1, 2].hist(df['num_medications'], bins=20, color=self.colors['secondary'], alpha=0.7)
        axes[1, 2].set_title('Number of Medications Distribution')
        axes[1, 2].set_xlabel('Number of Medications')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Data overview plot saved to {save_path}")
        
        plt.show()
    
    def plot_treatment_effects(
        self, 
        estimates: Dict[str, CausalEstimate], 
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot treatment effect estimates with confidence intervals.
        
        Args:
            estimates: Dictionary of causal estimates
            save_path: Path to save the figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        methods = list(estimates.keys())
        coefficients = [est.coefficient for est in estimates.values()]
        ci_lower = [est.ci_lower for est in estimates.values()]
        ci_upper = [est.ci_upper for est in estimates.values()]
        
        # Calculate error bars
        errors_lower = [coef - lower for coef, lower in zip(coefficients, ci_lower)]
        errors_upper = [upper - coef for coef, upper in zip(coefficients, ci_upper)]
        
        y_pos = np.arange(len(methods))
        
        # Create horizontal error bar plot
        ax.errorbar(coefficients, y_pos, xerr=[errors_lower, errors_upper], 
                   fmt='o', markersize=8, capsize=5, capthick=2,
                   color=self.colors['primary'], ecolor=self.colors['dark_gray'])
        
        # Add vertical line at x=0
        ax.axvline(x=0, color=self.colors['neutral'], linestyle='--', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods)
        ax.set_xlabel('Treatment Effect (Coefficient)')
        ax.set_title('Causal Treatment Effect Estimates\nImpact of A1C Testing on 30-day Readmission', 
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add significance indicators
        for i, (method, est) in enumerate(estimates.items()):
            if est.is_significant:
                ax.text(est.coefficient + 0.001, i, '*', fontsize=20, 
                       color=self.colors['neutral'], fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Treatment effects plot saved to {save_path}")
        
        plt.show()
    
    def plot_readmission_by_subgroups(
        self, 
        df: pd.DataFrame, 
        subgroup_var: str,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot readmission rates by subgroups and testing status.
        
        Args:
            df: Dataset
            subgroup_var: Variable to use for subgrouping
            save_path: Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calculate readmission rates
        subgroup_rates = df.groupby([subgroup_var, 'A1C_tested'])['readmitted'].mean().unstack()
        
        # Plot 1: Readmission rates by subgroup and testing status
        subgroup_rates.plot(kind='bar', ax=ax1, color=[self.colors['light_gray'], self.colors['primary']])
        ax1.set_title(f'Readmission Rates by {subgroup_var}')
        ax1.set_xlabel(subgroup_var)
        ax1.set_ylabel('Readmission Rate')
        ax1.legend(['Not Tested', 'A1C Tested'])
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: A1C testing rates by subgroup
        testing_rates = df.groupby(subgroup_var)['A1C_tested'].mean()
        testing_rates.plot(kind='bar', ax=ax2, color=self.colors['accent'])
        ax2.set_title(f'A1C Testing Rates by {subgroup_var}')
        ax2.set_xlabel(subgroup_var)
        ax2.set_ylabel('A1C Testing Rate')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Subgroup analysis plot saved to {save_path}")
        
        plt.show()
    
    def plot_heterogeneous_effects(
        self, 
        heterogeneous_results: Dict[str, Dict[str, CausalEstimate]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot heterogeneous treatment effects across subgroups.
        
        Args:
            heterogeneous_results: Results from heterogeneous effects analysis
            save_path: Path to save the figure
        """
        n_vars = len(heterogeneous_results)
        fig, axes = plt.subplots(1, n_vars, figsize=(6*n_vars, 6))
        
        if n_vars == 1:
            axes = [axes]
        
        for i, (var, results) in enumerate(heterogeneous_results.items()):
            subgroups = list(results.keys())
            coefficients = [est.coefficient for est in results.values()]
            ci_lower = [est.ci_lower for est in results.values()]
            ci_upper = [est.ci_upper for est in results.values()]
            
            # Calculate error bars
            errors_lower = [coef - lower for coef, lower in zip(coefficients, ci_lower)]
            errors_upper = [upper - coef for coef, upper in zip(coefficients, ci_upper)]
            
            y_pos = np.arange(len(subgroups))
            
            axes[i].errorbar(coefficients, y_pos, xerr=[errors_lower, errors_upper],
                           fmt='o', markersize=8, capsize=5, capthick=2,
                           color=self.colors['secondary'], ecolor=self.colors['dark_gray'])
            
            axes[i].axvline(x=0, color=self.colors['neutral'], linestyle='--', alpha=0.7)
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(subgroups)
            axes[i].set_xlabel('Treatment Effect')
            axes[i].set_title(f'Heterogeneous Effects by {var}')
            axes[i].grid(True, alpha=0.3)
            
            # Add significance indicators
            for j, (subgroup, est) in enumerate(results.items()):
                if est.is_significant:
                    axes[i].text(est.coefficient + 0.001, j, '*', fontsize=16, 
                               color=self.colors['neutral'], fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heterogeneous effects plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_performance(
        self, 
        performance: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot model performance metrics.
        
        Args:
            performance: Performance metrics from model evaluation
            save_path: Path to save the figure
        """
        methods = list(performance.keys())
        metrics = ['ml_g0_logloss', 'ml_g1_logloss', 'ml_m_logloss']
        metric_labels = ['Outcome Model (Control)', 'Outcome Model (Treated)', 'Treatment Model']
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.arange(len(methods))
        width = 0.25
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = []
            for method in methods:
                if 'error' not in performance[method]:
                    values.append(performance[method].get(metric, np.nan))
                else:
                    values.append(np.nan)
            
            ax.bar(x + i*width, values, width, label=label, alpha=0.8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Log Loss')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model performance plot saved to {save_path}")
        
        plt.show()
    
    def create_correlation_heatmap(
        self, 
        df: pd.DataFrame, 
        variables: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create correlation heatmap for selected variables.
        
        Args:
            df: Dataset
            variables: Variables to include in correlation matrix
            save_path: Path to save the figure
        """
        correlation_matrix = df[variables].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        
        ax.set_title('Correlation Matrix of Key Variables', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation heatmap saved to {save_path}")
        
        plt.show()
    
    def plot_causal_dag(self, save_path: Optional[str] = None) -> None:
        """
        Create a directed acyclic graph showing causal assumptions.
        
        Args:
            save_path: Path to save the figure
        """
        import networkx as nx
        
        # Create DAG
        G = nx.DiGraph()
        
        # Add nodes
        nodes = {
            'Demographics': (0, 2),
            'Medical History': (0, 1),
            'Disease Severity': (0, 0),
            'A1C Testing': (2, 1),
            'Medication Change': (3, 0.5),
            'Readmission': (4, 1)
        }
        
        for node, pos in nodes.items():
            G.add_node(node, pos=pos)
        
        # Add edges
        edges = [
            ('Demographics', 'A1C Testing'),
            ('Medical History', 'A1C Testing'),
            ('Disease Severity', 'A1C Testing'),
            ('Demographics', 'Readmission'),
            ('Medical History', 'Readmission'),
            ('Disease Severity', 'Readmission'),
            ('A1C Testing', 'Medication Change'),
            ('A1C Testing', 'Readmission'),
            ('Medication Change', 'Readmission')
        ]
        
        G.add_edges_from(edges)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        pos = nx.get_node_attributes(G, 'pos')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=self.colors['primary'], 
                              node_size=3000, alpha=0.9, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color=self.colors['dark_gray'],
                              arrows=True, arrowsize=20, alpha=0.7, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        
        # Highlight treatment and outcome
        nx.draw_networkx_nodes(G, pos, nodelist=['A1C Testing'], 
                              node_color=self.colors['accent'], 
                              node_size=3000, alpha=0.9, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=['Readmission'], 
                              node_color=self.colors['neutral'], 
                              node_size=3000, alpha=0.9, ax=ax)
        
        ax.set_title('Causal Directed Acyclic Graph (DAG)', fontweight='bold', fontsize=14)
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['primary'], 
                      markersize=15, label='Confounders'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['accent'], 
                      markersize=15, label='Treatment'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['neutral'], 
                      markersize=15, label='Outcome')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Causal DAG saved to {save_path}")
        
        plt.show()