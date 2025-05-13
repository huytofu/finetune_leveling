"""
Data Flow Pipeline Module.

This module provides a streamlined pipeline for:
1. Data extraction from various sources
2. Quality and distribution checks
3. Automated decision making for fine-tuning
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import pymongo
from cassandra.cluster import Cluster
from elasticsearch import Elasticsearch
from pymongo import MongoClient

from .data_monitoring import DataMonitor, DataQualityConfig
from .pipelines import FineTunePipeline, FineTuneConfig
from .error_handler import ErrorHandler
from .nlp_monitoring import AdvancedNLPMonitor

logger = logging.getLogger(__name__)

@dataclass
class DataSourceConfig:
    """Configuration for data source connections."""
    source_type: str  # "mongodb", "cassandra", "elasticsearch", "local"
    
    # MongoDB settings
    mongodb_uri: Optional[str] = None
    mongodb_db: Optional[str] = None
    mongodb_collection: Optional[str] = None
    
    # Cassandra settings
    cassandra_hosts: Optional[List[str]] = None
    cassandra_keyspace: Optional[str] = None
    cassandra_table: Optional[str] = None
    
    # Elasticsearch settings
    es_hosts: Optional[List[str]] = None
    es_index: Optional[str] = None
    
    # Local file settings
    local_path: Optional[str] = None
    file_format: Optional[str] = None  # "csv", "json", "parquet"

@dataclass
class QualityCheckConfig:
    """Configuration for quality checks and thresholds."""
    drift_threshold: float = 0.15  # KL divergence threshold (lower is better)
    quality_threshold: float = 0.8  # Quality score threshold (higher is better)
    min_samples: int = 1000  # Minimum number of samples required
    output_path: str = "quality_report.json"
    
    # Advanced NLP thresholds
    max_perplexity: float = 100.0  # Maximum acceptable perplexity score
    min_coherence: float = 0.4  # Minimum acceptable coherence score
    max_topic_drift: float = 0.3  # Maximum acceptable topic drift
    max_semantic_drift: float = 0.5  # Maximum acceptable semantic drift

class DataFlowPipeline:
    """
    End-to-end pipeline for data processing and quality checks.
    
    This pipeline:
    1. Loads data from various sources
    2. Performs quality and distribution checks
    3. Makes automated decisions for fine-tuning
    """
    
    def __init__(
        self,
        data_source_config: DataSourceConfig,
        quality_config: QualityCheckConfig,
        data_monitor: Optional[DataMonitor] = None,
        nlp_monitor: Optional[AdvancedNLPMonitor] = None,
        finetune_config: Optional[FineTuneConfig] = None
    ):
        self.data_source_config = data_source_config
        self.quality_config = quality_config
        
        # Initialize monitors
        if data_monitor is None:
            data_monitor = DataMonitor(DataQualityConfig())
        self.data_monitor = data_monitor
        
        if nlp_monitor is None:
            nlp_monitor = AdvancedNLPMonitor()
        self.nlp_monitor = nlp_monitor
        
        self.finetune_config = finetune_config
        
    def load_data(self) -> pd.DataFrame:
        """Load data from the configured source."""
        try:
            if self.data_source_config.source_type == "mongodb":
                return self._load_from_mongodb()
            elif self.data_source_config.source_type == "cassandra":
                return self._load_from_cassandra()
            elif self.data_source_config.source_type == "elasticsearch":
                return self._load_from_elasticsearch()
            elif self.data_source_config.source_type == "local":
                return self._load_from_local()
            else:
                raise ValueError(f"Unsupported source type: {self.data_source_config.source_type}")
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            ErrorHandler.handle_error(e, "load_data")
            raise
            
    def _load_from_mongodb(self) -> pd.DataFrame:
        """Load data from MongoDB."""
        client = MongoClient(self.data_source_config.mongodb_uri)
        db = client[self.data_source_config.mongodb_db]
        collection = db[self.data_source_config.mongodb_collection]
        
        # Convert MongoDB cursor to DataFrame
        data = pd.DataFrame(list(collection.find()))
        return data
        
    def _load_from_cassandra(self) -> pd.DataFrame:
        """Load data from Cassandra."""
        cluster = Cluster(self.data_source_config.cassandra_hosts)
        session = cluster.connect(self.data_source_config.cassandra_keyspace)
        
        # Execute query and fetch all rows
        query = f"SELECT * FROM {self.data_source_config.cassandra_table}"
        rows = session.execute(query)
        
        # Convert to DataFrame
        data = pd.DataFrame(list(rows))
        return data
        
    def _load_from_elasticsearch(self) -> pd.DataFrame:
        """Load data from Elasticsearch."""
        es = Elasticsearch(self.data_source_config.es_hosts)
        
        # Search query to fetch all documents
        query = {"query": {"match_all": {}}}
        response = es.search(
            index=self.data_source_config.es_index,
            body=query,
            size=10000  # Adjust based on your needs
        )
        
        # Extract documents and convert to DataFrame
        documents = [hit["_source"] for hit in response["hits"]["hits"]]
        data = pd.DataFrame(documents)
        return data
        
    def _load_from_local(self) -> pd.DataFrame:
        """Load data from local file."""
        path = Path(self.data_source_config.local_path)
        
        if self.data_source_config.file_format == "csv":
            return pd.read_csv(path)
        elif self.data_source_config.file_format == "json":
            return pd.read_json(path)
        elif self.data_source_config.file_format == "parquet":
            return pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_source_config.file_format}")
            
    def check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive data quality checks including advanced NLP metrics.
        
        Returns:
            Dictionary containing quality metrics and pass/fail decision
        """
        try:
            # Basic quality checks
            if len(data) < self.quality_config.min_samples:
                return {
                    "status": "fail",
                    "reason": f"Insufficient samples: {len(data)} < {self.quality_config.min_samples}",
                    "metrics": {}
                }
                
            # Get text column
            text_column = "text"  # Adjust if your text column has a different name
            texts = data[text_column].tolist()
            
            # Basic quality metrics
            basic_metrics = self.data_monitor.validate_data_quality(data)
            quality_score = self._compute_quality_score(
                basic_metrics.get("custom_checks", {})
            )
            
            # Advanced NLP metrics
            nlp_metrics = {
                "perplexity": self.nlp_monitor.compute_perplexity(texts),
                "coherence": self.nlp_monitor.compute_coherence(texts)
            }
            
            # Distribution and drift analysis
            if hasattr(self.data_monitor, "train_stats") and self.data_monitor.train_stats:
                reference_texts = self.data_monitor.train_stats.get("reference_texts", [])
                if reference_texts:
                    nlp_metrics.update({
                        "topic_distribution": self.nlp_monitor.monitor_topic_distribution(
                            reference_texts,
                            texts
                        ),
                        "semantic_drift": self.nlp_monitor.detect_semantic_drift(
                            reference_texts,
                            texts
                        )
                    })
                    
            # Make pass/fail decision
            status = "pass"
            reasons = []
            
            # Check basic quality
            if quality_score < self.quality_config.quality_threshold:
                status = "fail"
                reasons.append(
                    f"Low quality score: {quality_score:.3f} < {self.quality_config.quality_threshold}"
                )
                
            # Check perplexity
            mean_perplexity = nlp_metrics["perplexity"].get("mean_perplexity", 0)
            if mean_perplexity > self.quality_config.max_perplexity:
                status = "fail"
                reasons.append(
                    f"High perplexity: {mean_perplexity:.1f} > {self.quality_config.max_perplexity}"
                )
                
            # Check coherence
            c_v_coherence = nlp_metrics["coherence"].get("c_v_coherence", 0)
            if c_v_coherence < self.quality_config.min_coherence:
                status = "fail"
                reasons.append(
                    f"Low coherence: {c_v_coherence:.3f} < {self.quality_config.min_coherence}"
                )
                
            # Check topic drift
            if "topic_distribution" in nlp_metrics:
                topic_drift = nlp_metrics["topic_distribution"].get("topic_drift_score", 0)
                if topic_drift > self.quality_config.max_topic_drift:
                    status = "fail"
                    reasons.append(
                        f"High topic drift: {topic_drift:.3f} > {self.quality_config.max_topic_drift}"
                    )
                    
            # Check semantic drift
            if "semantic_drift" in nlp_metrics:
                semantic_drift = nlp_metrics["semantic_drift"].get("semantic_drift_score", 0)
                if semantic_drift > self.quality_config.max_semantic_drift:
                    status = "fail"
                    reasons.append(
                        f"High semantic drift: {semantic_drift:.3f} > {self.quality_config.max_semantic_drift}"
                    )
                    
            # Prepare report
            report = {
                "status": status,
                "reason": "; ".join(reasons) if reasons else "All checks passed",
                "metrics": {
                    "basic_quality": {
                        "score": quality_score,
                        "details": basic_metrics
                    },
                    "nlp_metrics": nlp_metrics
                }
            }
            
            # Save report
            with open(self.quality_config.output_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            return report
            
        except Exception as e:
            logger.error(f"Error checking data quality: {str(e)}")
            ErrorHandler.handle_error(e, "check_data_quality")
            return {"status": "fail", "reason": str(e), "metrics": {}}
            
    def _compute_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Compute overall quality score from various metrics."""
        scores = []
        
        # Score based on missing values
        if "missing_values" in metrics:
            missing_ratio = max(metrics["missing_values"]["missing_ratio"].values())
            scores.append(1 - missing_ratio)
            
        # Score based on duplicates
        if "duplicates" in metrics:
            duplicate_ratio = metrics["duplicates"]["duplicate_ratio"]
            scores.append(1 - duplicate_ratio)
            
        # Score based on text quality
        if "text_quality" in metrics:
            text_stats = metrics["text_quality"]
            if "empty_texts" in text_stats:
                empty_ratio = text_stats["empty_texts"] / text_stats["length_stats"]["mean"]
                scores.append(1 - empty_ratio)
                
            if "special_chars" in text_stats:
                special_char_ratio = text_stats["special_chars"]
                scores.append(1 - special_char_ratio)
                
        # Return average score if we have any metrics
        return sum(scores) / len(scores) if scores else 0.0
        
    def run(self) -> bool:
        """
        Run the complete pipeline.
        
        Returns:
            Boolean indicating whether the pipeline completed successfully
        """
        try:
            # Step 1: Load data
            logger.info("Loading data...")
            data = self.load_data()
            
            # Step 2: Check quality
            logger.info("Checking data quality...")
            quality_report = self.check_data_quality(data)
            
            # Step 3: Decide whether to proceed
            if quality_report["status"] == "fail":
                logger.warning(f"Data quality check failed: {quality_report['reason']}")
                return False
                
            # Step 4: Proceed with fine-tuning if configured
            if self.finetune_config and quality_report["status"] == "pass":
                logger.info("Starting fine-tuning...")
                pipeline = FineTunePipeline(self.finetune_config)
                pipeline.train(data)
                
            return True
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            ErrorHandler.handle_error(e, "run_pipeline")
            return False 