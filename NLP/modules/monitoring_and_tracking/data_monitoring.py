"""
Data Quality and Distribution Monitoring Module.

This module provides comprehensive data quality monitoring, distribution analysis,
and drift detection for NLP datasets. It integrates multiple state-of-the-art
tools and custom implementations for thorough data validation and monitoring.
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

import great_expectations as ge
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from soda.scan import Scan
import torch
from torch.utils.data import Dataset
import mlflow

from .error_handler import ErrorHandler

logger = logging.getLogger(__name__)

@dataclass
class DataQualityConfig:
    """Configuration for data quality monitoring."""
    # Basic checks
    check_missing_values: bool = True
    check_duplicates: bool = True
    check_data_types: bool = True
    
    # Distribution checks
    check_distribution: bool = True
    distribution_threshold: float = 0.05  # KL divergence threshold
    
    # Text-specific checks
    min_text_length: int = 1
    max_text_length: int = 512
    check_special_chars: bool = True
    language_check: bool = True
    
    # Drift detection
    drift_detection: bool = True
    drift_window_size: int = 1000
    drift_threshold: float = 0.1
    
    # Validation thresholds
    missing_threshold: float = 0.1
    duplicate_threshold: float = 0.05
    outlier_threshold: float = 3.0
    
    # Monitoring frequency
    check_frequency: str = "batch"  # "batch" or "sample"
    sample_size: int = 1000

class DataMonitor:
    """
    Comprehensive data quality and distribution monitoring system.
    
    This class provides:
    1. Data quality validation using Great Expectations
    2. Distribution monitoring and drift detection
    3. Text-specific quality checks
    4. Integration with MLflow for tracking
    5. Real-time monitoring during training
    
    Attributes:
        config: Configuration for data monitoring
        ge_context: Great Expectations context
        soda_scan: Soda Core scan object
        train_stats: Statistics from training data
        current_stats: Current monitoring statistics
        drift_history: History of drift measurements
    """
    
    def __init__(
        self,
        config: DataQualityConfig,
        mlflow_tracking: bool = True
    ):
        self.config = config
        self.mlflow_tracking = mlflow_tracking
        
        # Initialize monitoring components
        self.ge_context = ge.get_context()
        self.soda_scan = Scan()
        
        # Initialize statistics storage
        self.train_stats = {}
        self.current_stats = {}
        self.drift_history = []
        
        # Initialize data quality expectations
        self._setup_expectations()
        
    def _setup_expectations(self):
        """Set up Great Expectations suite for data validation."""
        self.expectation_suite = self.ge_context.create_expectation_suite(
            suite_name="nlp_data_suite"
        )
        
        # Add basic expectations
        if self.config.check_missing_values:
            self.expectation_suite.add_expectation(
                ge.core.ExpectationConfiguration(
                    expectation_type="expect_column_values_to_not_be_null",
                    kwargs={"column": "text"}
                )
            )
            
        if self.config.check_duplicates:
            self.expectation_suite.add_expectation(
                ge.core.ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_unique",
                    kwargs={"column": "text"}
                )
            )
            
        # Add text-specific expectations
        if self.config.check_special_chars:
            self.expectation_suite.add_expectation(
                ge.core.ExpectationConfiguration(
                    expectation_type="expect_column_values_to_match_regex",
                    kwargs={
                        "column": "text",
                        "regex": r"^[a-zA-Z0-9\s\.,!?'-]+$"
                    }
                )
            )
            
    def compute_distribution_stats(
        self,
        dataset: Union[Dataset, pd.DataFrame],
        compute_full: bool = True
    ) -> Dict[str, Any]:
        """
        Compute comprehensive distribution statistics for the dataset.
        
        Args:
            dataset: Input dataset
            compute_full: Whether to compute full statistics
            
        Returns:
            Dictionary containing distribution statistics
        """
        try:
            # Convert to pandas if needed
            if isinstance(dataset, Dataset):
                data = pd.DataFrame(dataset)
            else:
                data = dataset
                
            stats = {}
            
            # Basic statistics
            stats["sample_size"] = len(data)
            stats["missing_ratio"] = data.isnull().mean().to_dict()
            stats["duplicate_ratio"] = data.duplicated().mean()
            
            # Text statistics
            if "text" in data.columns:
                text_lengths = data["text"].str.len()
                stats["text_length"] = {
                    "mean": text_lengths.mean(),
                    "std": text_lengths.std(),
                    "min": text_lengths.min(),
                    "max": text_lengths.max(),
                    "quantiles": text_lengths.quantile([0.25, 0.5, 0.75]).to_dict()
                }
                
            # Full distribution statistics
            if compute_full:
                for column in data.select_dtypes(include=[np.number]).columns:
                    col_stats = {
                        "mean": data[column].mean(),
                        "std": data[column].std(),
                        "skew": stats.skew(data[column].dropna()),
                        "kurtosis": stats.kurtosis(data[column].dropna()),
                        "histogram": np.histogram(data[column].dropna(), bins=50)
                    }
                    stats[f"{column}_distribution"] = col_stats
                    
            return stats
            
        except Exception as e:
            logger.error(f"Error computing distribution stats: {str(e)}")
            ErrorHandler.handle_error(e, "compute_distribution_stats")
            return {}
            
    def detect_drift(
        self,
        train_data: Union[Dataset, pd.DataFrame],
        current_data: Union[Dataset, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Detect distribution drift between training and current data.
        
        Args:
            train_data: Training dataset
            current_data: Current dataset
            
        Returns:
            Dictionary containing drift metrics
        """
        try:
            # Convert to pandas if needed
            if isinstance(train_data, Dataset):
                train_df = pd.DataFrame(train_data)
            else:
                train_df = train_data
                
            if isinstance(current_data, Dataset):
                current_df = pd.DataFrame(current_data)
            else:
                current_df = current_data
                
            drift_metrics = {}
            
            # KL divergence for numerical columns
            for column in train_df.select_dtypes(include=[np.number]).columns:
                # Normalize data
                scaler = StandardScaler()
                train_normalized = scaler.fit_transform(train_df[column].values.reshape(-1, 1))
                current_normalized = scaler.transform(current_df[column].values.reshape(-1, 1))
                
                # Compute histograms
                train_hist, _ = np.histogram(train_normalized, bins=50, density=True)
                current_hist, _ = np.histogram(current_normalized, bins=50, density=True)
                
                # Add small constant to avoid division by zero
                train_hist = train_hist + 1e-10
                current_hist = current_hist + 1e-10
                
                # Compute KL divergence
                kl_div = stats.entropy(train_hist, current_hist)
                drift_metrics[f"{column}_drift"] = kl_div
                
            # Text-specific drift detection
            if "text" in train_df.columns:
                # Compare length distributions
                train_lengths = train_df["text"].str.len()
                current_lengths = current_df["text"].str.len()
                
                length_ks_stat = stats.ks_2samp(train_lengths, current_lengths).statistic
                drift_metrics["text_length_drift"] = length_ks_stat
                
            # Log drift metrics if MLflow is enabled
            if self.mlflow_tracking:
                mlflow.log_metrics(
                    {f"drift_{k}": v for k, v in drift_metrics.items()}
                )
                
            return drift_metrics
            
        except Exception as e:
            logger.error(f"Error detecting drift: {str(e)}")
            ErrorHandler.handle_error(e, "detect_drift")
            return {}
            
    def validate_data_quality(
        self,
        dataset: Union[Dataset, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive data quality validation.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dictionary containing validation results
        """
        try:
            # Convert to pandas if needed
            if isinstance(dataset, Dataset):
                data = pd.DataFrame(dataset)
            else:
                data = dataset
                
            validation_results = {}
            
            # Run Great Expectations validation
            ge_results = self.ge_context.run_validation(
                data,
                expectation_suite_name="nlp_data_suite"
            )
            validation_results["expectations"] = ge_results.to_json_dict()
            
            # Run Soda Core checks
            soda_results = self.soda_scan.scan(data)
            validation_results["soda_checks"] = soda_results
            
            # Custom validations
            custom_checks = {
                "missing_values": self._check_missing_values(data),
                "duplicates": self._check_duplicates(data),
                "text_quality": self._check_text_quality(data),
                "data_types": self._check_data_types(data)
            }
            validation_results["custom_checks"] = custom_checks
            
            # Log validation results if MLflow is enabled
            if self.mlflow_tracking:
                mlflow.log_dict(
                    validation_results,
                    "data_validation_results.json"
                )
                
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating data quality: {str(e)}")
            ErrorHandler.handle_error(e, "validate_data_quality")
            return {}
            
    def _check_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values in the dataset."""
        missing_stats = {
            "total_missing": data.isnull().sum().to_dict(),
            "missing_ratio": data.isnull().mean().to_dict()
        }
        return missing_stats
        
    def _check_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate entries in the dataset."""
        duplicate_stats = {
            "total_duplicates": data.duplicated().sum(),
            "duplicate_ratio": data.duplicated().mean(),
            "duplicate_indices": data[data.duplicated()].index.tolist()
        }
        return duplicate_stats
        
    def _check_text_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform text-specific quality checks."""
        if "text" not in data.columns:
            return {}
            
        text_stats = {
            "length_stats": {
                "mean": data["text"].str.len().mean(),
                "std": data["text"].str.len().std(),
                "min": data["text"].str.len().min(),
                "max": data["text"].str.len().max()
            },
            "special_chars": data["text"].str.contains(r'[^a-zA-Z0-9\s\.,!?-]').mean(),
            "empty_texts": (data["text"].str.strip() == "").sum()
        }
        return text_stats
        
    def _check_data_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check data types and their consistency."""
        type_stats = {
            "dtypes": data.dtypes.astype(str).to_dict(),
            "type_consistency": {
                col: len(data[col].dropna().unique()) 
                for col in data.columns
            }
        }
        return type_stats
        
    def monitor_training_data(
        self,
        batch: Dict[str, torch.Tensor],
        step: int
    ) -> Dict[str, Any]:
        """
        Monitor data quality during training.
        
        Args:
            batch: Current training batch
            step: Current training step
            
        Returns:
            Dictionary containing monitoring results
        """
        try:
            monitoring_results = {}
            
            # Convert batch to pandas
            batch_df = pd.DataFrame({
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            })
            
            # Compute quick statistics
            monitoring_results["batch_stats"] = {
                "size": len(batch_df),
                "missing_values": batch_df.isnull().sum().to_dict(),
                "value_ranges": {
                    col: {
                        "min": batch_df[col].min(),
                        "max": batch_df[col].max()
                    }
                    for col in batch_df.select_dtypes(include=[np.number]).columns
                }
            }
            
            # Check for drift if enough steps have passed
            if step % self.config.drift_window_size == 0 and self.train_stats:
                drift_metrics = self.detect_drift(
                    pd.DataFrame(self.train_stats["reference_batch"]),
                    batch_df
                )
                monitoring_results["drift_metrics"] = drift_metrics
                
            # Store current batch stats
            self.current_stats[step] = monitoring_results
            
            # Log monitoring results if MLflow is enabled
            if self.mlflow_tracking:
                mlflow.log_metrics(
                    {f"monitoring_{k}": v 
                     for k, v in monitoring_results["batch_stats"].items()
                     if isinstance(v, (int, float))},
                    step=step
                )
                
            return monitoring_results
            
        except Exception as e:
            logger.error(f"Error monitoring training data: {str(e)}")
            ErrorHandler.handle_error(e, "monitor_training_data")
            return {} 