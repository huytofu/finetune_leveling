import mlflow
import os
import time
import psutil
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import asdict
import sqlite3
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MLflowTracker:
    """Handles MLflow tracking for finetuning experiments."""
    
    def __init__(self, experiment_name: str, tracking_uri: str = "sqlite:///mlflow.db"):
        """Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: URI for MLflow tracking (default: SQLite)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.start_time = None
        self.run_id = None
        
        # Setup MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        # Create SQLite database for additional metadata
        self._setup_database()
        
    def _setup_database(self):
        """Setup SQLite database for additional metadata."""
        try:
            db_path = Path(self.tracking_uri.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create tables for different types of metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_metadata (
                    run_id TEXT PRIMARY KEY,
                    experiment_name TEXT,
                    task_type TEXT,
                    data_source TEXT,
                    dataset_size INTEGER,
                    num_classes INTEGER,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS resource_metrics (
                    run_id TEXT,
                    timestamp TIMESTAMP,
                    cpu_percent REAL,
                    memory_percent REAL,
                    gpu_percent REAL,
                    gpu_memory_percent REAL,
                    FOREIGN KEY (run_id) REFERENCES experiment_metadata(run_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS timing_metrics (
                    run_id TEXT,
                    phase TEXT,
                    duration REAL,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES experiment_metadata(run_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    run_id TEXT,
                    artifact_type TEXT,
                    artifact_path TEXT,
                    artifact_size INTEGER,
                    created_at TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES experiment_metadata(run_id)
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            raise
    
    def start_run(self, run_name: Optional[str] = None):
        """Start a new MLflow run."""
        try:
            self.start_time = time.time()
            self.run = mlflow.start_run(run_name=run_name)
            self.run_id = self.run.info.run_id
            
            # Log initial metadata
            mlflow.log_param("start_time", datetime.now().isoformat())
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            raise
    
    def end_run(self):
        """End the current MLflow run."""
        try:
            if self.run:
                duration = time.time() - self.start_time
                mlflow.log_metric("total_duration", duration)
                mlflow.log_param("end_time", datetime.now().isoformat())
                mlflow.end_run()
                
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")
            raise
    
    def log_experiment_metadata(self, metadata: Dict[str, Any]):
        """Log experiment metadata to both MLflow and SQLite."""
        try:
            # Log to MLflow
            for key, value in metadata.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
                else:
                    mlflow.log_param(key, str(value))
            
            # Log to SQLite
            conn = sqlite3.connect(self.tracking_uri.replace("sqlite:///", ""))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO experiment_metadata 
                (run_id, experiment_name, task_type, data_source, dataset_size, 
                 num_classes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.run_id,
                self.experiment_name,
                metadata.get("task_type"),
                metadata.get("data_source"),
                metadata.get("dataset_size"),
                metadata.get("num_classes"),
                datetime.now(),
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log experiment metadata: {e}")
            raise
    
    def log_resource_metrics(self):
        """Log current resource usage metrics."""
        try:
            metrics = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent
            }
            
            # Log GPU metrics if available
            if torch.cuda.is_available():
                metrics.update({
                    "gpu_percent": torch.cuda.utilization(),
                    "gpu_memory_percent": torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                })
            
            # Log to MLflow
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log to SQLite
            conn = sqlite3.connect(self.tracking_uri.replace("sqlite:///", ""))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO resource_metrics 
                (run_id, timestamp, cpu_percent, memory_percent, gpu_percent, gpu_memory_percent)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                self.run_id,
                datetime.now(),
                metrics.get("cpu_percent"),
                metrics.get("memory_percent"),
                metrics.get("gpu_percent"),
                metrics.get("gpu_memory_percent")
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log resource metrics: {e}")
            raise
    
    def log_timing_metric(self, phase: str, duration: float):
        """Log timing metric for a specific phase."""
        try:
            # Log to MLflow
            mlflow.log_metric(f"{phase}_duration", duration)
            
            # Log to SQLite
            conn = sqlite3.connect(self.tracking_uri.replace("sqlite:///", ""))
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO timing_metrics (run_id, phase, duration, timestamp)
                VALUES (?, ?, ?, ?)
            """, (self.run_id, phase, duration, datetime.now()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log timing metric: {e}")
            raise
    
    def log_artifact(self, artifact_path: str, artifact_type: str):
        """Log an artifact to MLflow and SQLite."""
        try:
            # Log to MLflow
            mlflow.log_artifact(artifact_path)
            
            # Log to SQLite
            conn = sqlite3.connect(self.tracking_uri.replace("sqlite:///", ""))
            cursor = conn.cursor()
            
            artifact_size = os.path.getsize(artifact_path)
            
            cursor.execute("""
                INSERT INTO artifacts (run_id, artifact_type, artifact_path, 
                                     artifact_size, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                self.run_id,
                artifact_type,
                artifact_path,
                artifact_size,
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
            raise
    
    def log_dataset_metrics(self, dataset_metrics: Dict[str, Any]):
        """Log dataset-specific metrics."""
        try:
            # Log to MLflow
            for key, value in dataset_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"dataset_{key}", value)
                else:
                    mlflow.log_param(f"dataset_{key}", str(value))
            
            # Update SQLite metadata
            conn = sqlite3.connect(self.tracking_uri.replace("sqlite:///", ""))
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE experiment_metadata 
                SET dataset_size = ?, updated_at = ?
                WHERE run_id = ?
            """, (
                dataset_metrics.get("size"),
                datetime.now(),
                self.run_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log dataset metrics: {e}")
            raise
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of the experiment."""
        try:
            conn = sqlite3.connect(self.tracking_uri.replace("sqlite:///", ""))
            cursor = conn.cursor()
            
            # Get experiment metadata
            cursor.execute("""
                SELECT * FROM experiment_metadata WHERE run_id = ?
            """, (self.run_id,))
            metadata = cursor.fetchone()
            
            # Get resource metrics
            cursor.execute("""
                SELECT AVG(cpu_percent), AVG(memory_percent), 
                       AVG(gpu_percent), AVG(gpu_memory_percent)
                FROM resource_metrics WHERE run_id = ?
            """, (self.run_id,))
            resources = cursor.fetchone()
            
            # Get timing metrics
            cursor.execute("""
                SELECT phase, duration FROM timing_metrics WHERE run_id = ?
            """, (self.run_id,))
            timings = cursor.fetchall()
            
            # Get artifacts
            cursor.execute("""
                SELECT artifact_type, artifact_path, artifact_size 
                FROM artifacts WHERE run_id = ?
            """, (self.run_id,))
            artifacts = cursor.fetchall()
            
            conn.close()
            
            return {
                "metadata": {
                    "run_id": metadata[0],
                    "experiment_name": metadata[1],
                    "task_type": metadata[2],
                    "data_source": metadata[3],
                    "dataset_size": metadata[4],
                    "num_classes": metadata[5],
                    "created_at": metadata[6],
                    "updated_at": metadata[7]
                },
                "resources": {
                    "avg_cpu_percent": resources[0],
                    "avg_memory_percent": resources[1],
                    "avg_gpu_percent": resources[2],
                    "avg_gpu_memory_percent": resources[3]
                },
                "timings": dict(timings),
                "artifacts": [
                    {
                        "type": a[0],
                        "path": a[1],
                        "size": a[2]
                    }
                    for a in artifacts
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get experiment summary: {e}")
            return {} 