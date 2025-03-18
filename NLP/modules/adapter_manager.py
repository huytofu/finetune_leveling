import os
import json
import time
import torch
import logging
import tempfile
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from peft import PeftModel
from transformers import PreTrainedModel
from huggingface_hub import snapshot_download, hf_hub_download
import boto3
from google.cloud import storage
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AdapterInfo:
    """
    Information about a loaded adapter
    """
    adapter_id: str
    adapter_name: str
    adapter_type: str  # 'lora', 'qlora', 'adapter', etc.
    source: str  # 'local', 'huggingface', 'aws', 'gcp'
    path: str
    metadata: Dict[str, Any]
    last_used: float
    is_loaded: bool = False


class MultiAdapterManager:
    """
    Manager for handling multiple adapters for a base model.
    Supports loading from disk or cloud storage, and switching between adapters at runtime.
    """
    
    def __init__(
        self, 
        base_model: PreTrainedModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: Optional[str] = None,
        max_adapters_in_memory: int = 5,
        aws_credentials: Optional[Dict[str, str]] = None,
        gcp_credentials: Optional[str] = None
    ):
        """
        Initialize the MultiAdapterManager
        
        Args:
            base_model: The base model to which adapters will be applied
            device: Device to load the adapters on
            cache_dir: Directory to cache downloaded adapters
            max_adapters_in_memory: Maximum number of adapters to keep in memory
            aws_credentials: Dictionary with AWS credentials (access_key, secret_key, region)
            gcp_credentials: Path to GCP credentials file
        """
        self.base_model = base_model
        self.device = device
        self.cache_dir = cache_dir or tempfile.mkdtemp()
        self.max_adapters_in_memory = max_adapters_in_memory
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set up cloud storage clients
        self.aws_credentials = aws_credentials
        self.gcp_credentials = gcp_credentials
        self._setup_cloud_clients()
        
        # Adapter tracking
        self.adapters: Dict[str, AdapterInfo] = {}
        self.current_adapter_id: Optional[str] = None
        self.current_adapter_model: Optional[PeftModel] = None
    
    def _setup_cloud_clients(self):
        """Set up clients for cloud storage"""
        # Set up AWS client if credentials provided
        self.s3_client = None
        if self.aws_credentials:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_credentials.get('access_key'),
                aws_secret_access_key=self.aws_credentials.get('secret_key'),
                region_name=self.aws_credentials.get('region', 'us-east-1')
            )
        
        # Set up GCP client if credentials provided
        self.gcs_client = None
        if self.gcp_credentials:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.gcp_credentials
            self.gcs_client = storage.Client()
    
    def register_adapter_from_local(
        self, 
        adapter_id: str,
        adapter_path: str,
        adapter_name: Optional[str] = None,
        adapter_type: str = "lora",
        metadata: Optional[Dict[str, Any]] = None
    ) -> AdapterInfo:
        """
        Register an adapter from a local path
        
        Args:
            adapter_id: Unique identifier for the adapter
            adapter_path: Path to the adapter files
            adapter_name: Human-readable name for the adapter
            adapter_type: Type of adapter ('lora', 'qlora', 'adapter', etc.)
            metadata: Additional metadata about the adapter
            
        Returns:
            AdapterInfo: Information about the registered adapter
        """
        if not os.path.exists(adapter_path):
            raise ValueError(f"Adapter path {adapter_path} does not exist")
        
        adapter_name = adapter_name or os.path.basename(adapter_path)
        metadata = metadata or {}
        
        adapter_info = AdapterInfo(
            adapter_id=adapter_id,
            adapter_name=adapter_name,
            adapter_type=adapter_type,
            source="local",
            path=adapter_path,
            metadata=metadata,
            last_used=time.time(),
            is_loaded=False
        )
        
        self.adapters[adapter_id] = adapter_info
        logger.info(f"Registered local adapter: {adapter_id} from {adapter_path}")
        
        return adapter_info
    
    def register_adapter_from_huggingface(
        self,
        adapter_id: str,
        repo_id: str,
        adapter_name: Optional[str] = None,
        adapter_type: str = "lora",
        metadata: Optional[Dict[str, Any]] = None,
        revision: str = "main",
        use_auth_token: Optional[str] = None
    ) -> AdapterInfo:
        """
        Register an adapter from Hugging Face Hub
        
        Args:
            adapter_id: Unique identifier for the adapter
            repo_id: Hugging Face repository ID
            adapter_name: Human-readable name for the adapter
            adapter_type: Type of adapter ('lora', 'qlora', 'adapter', etc.)
            metadata: Additional metadata about the adapter
            revision: Git revision to use
            use_auth_token: Hugging Face auth token for private repositories
            
        Returns:
            AdapterInfo: Information about the registered adapter
        """
        adapter_name = adapter_name or repo_id.split('/')[-1]
        metadata = metadata or {}
        metadata['repo_id'] = repo_id
        metadata['revision'] = revision
        
        # Create a dedicated cache directory for this adapter
        adapter_cache_dir = os.path.join(self.cache_dir, f"hf_{adapter_id}")
        os.makedirs(adapter_cache_dir, exist_ok=True)
        
        adapter_info = AdapterInfo(
            adapter_id=adapter_id,
            adapter_name=adapter_name,
            adapter_type=adapter_type,
            source="huggingface",
            path=adapter_cache_dir,
            metadata=metadata,
            last_used=time.time(),
            is_loaded=False
        )
        
        self.adapters[adapter_id] = adapter_info
        logger.info(f"Registered Hugging Face adapter: {adapter_id} from {repo_id}")
        
        return adapter_info
    
    def register_adapter_from_aws(
        self,
        adapter_id: str,
        bucket: str,
        prefix: str,
        adapter_name: Optional[str] = None,
        adapter_type: str = "lora",
        metadata: Optional[Dict[str, Any]] = None
    ) -> AdapterInfo:
        """
        Register an adapter from AWS S3
        
        Args:
            adapter_id: Unique identifier for the adapter
            bucket: S3 bucket name
            prefix: S3 prefix for the adapter files
            adapter_name: Human-readable name for the adapter
            adapter_type: Type of adapter ('lora', 'qlora', 'adapter', etc.)
            metadata: Additional metadata about the adapter
            
        Returns:
            AdapterInfo: Information about the registered adapter
        """
        if not self.s3_client:
            raise ValueError("AWS credentials not provided")
        
        adapter_name = adapter_name or f"{bucket}/{prefix}"
        metadata = metadata or {}
        metadata['bucket'] = bucket
        metadata['prefix'] = prefix
        
        # Create a dedicated cache directory for this adapter
        adapter_cache_dir = os.path.join(self.cache_dir, f"aws_{adapter_id}")
        os.makedirs(adapter_cache_dir, exist_ok=True)
        
        adapter_info = AdapterInfo(
            adapter_id=adapter_id,
            adapter_name=adapter_name,
            adapter_type=adapter_type,
            source="aws",
            path=adapter_cache_dir,
            metadata=metadata,
            last_used=time.time(),
            is_loaded=False
        )
        
        self.adapters[adapter_id] = adapter_info
        logger.info(f"Registered AWS S3 adapter: {adapter_id} from {bucket}/{prefix}")
        
        return adapter_info
    
    def register_adapter_from_gcp(
        self,
        adapter_id: str,
        bucket: str,
        prefix: str,
        adapter_name: Optional[str] = None,
        adapter_type: str = "lora",
        metadata: Optional[Dict[str, Any]] = None
    ) -> AdapterInfo:
        """
        Register an adapter from GCP Storage
        
        Args:
            adapter_id: Unique identifier for the adapter
            bucket: GCS bucket name
            prefix: GCS prefix for the adapter files
            adapter_name: Human-readable name for the adapter
            adapter_type: Type of adapter ('lora', 'qlora', 'adapter', etc.)
            metadata: Additional metadata about the adapter
            
        Returns:
            AdapterInfo: Information about the registered adapter
        """
        if not self.gcs_client:
            raise ValueError("GCP credentials not provided")
        
        adapter_name = adapter_name or f"{bucket}/{prefix}"
        metadata = metadata or {}
        metadata['bucket'] = bucket
        metadata['prefix'] = prefix
        
        # Create a dedicated cache directory for this adapter
        adapter_cache_dir = os.path.join(self.cache_dir, f"gcp_{adapter_id}")
        os.makedirs(adapter_cache_dir, exist_ok=True)
        
        adapter_info = AdapterInfo(
            adapter_id=adapter_id,
            adapter_name=adapter_name,
            adapter_type=adapter_type,
            source="gcp",
            path=adapter_cache_dir,
            metadata=metadata,
            last_used=time.time(),
            is_loaded=False
        )
        
        self.adapters[adapter_id] = adapter_info
        logger.info(f"Registered GCP Storage adapter: {adapter_id} from {bucket}/{prefix}")
        
        return adapter_info
    
    def _download_adapter_if_needed(self, adapter_id: str) -> str:
        """
        Download adapter from source if not already downloaded
        
        Args:
            adapter_id: Identifier for the adapter
            
        Returns:
            str: Path to the downloaded adapter
        """
        if adapter_id not in self.adapters:
            raise ValueError(f"Adapter {adapter_id} not registered")
        
        adapter_info = self.adapters[adapter_id]
        
        # If already downloaded, return the path
        if adapter_info.source == "local":
            return adapter_info.path
        
        # Check if adapter has already been downloaded (look for adapter_config.json)
        config_path = os.path.join(adapter_info.path, "adapter_config.json")
        if os.path.exists(config_path):
            logger.info(f"Adapter {adapter_id} already downloaded to {adapter_info.path}")
            return adapter_info.path
        
        # Download based on source
        if adapter_info.source == "huggingface":
            repo_id = adapter_info.metadata.get('repo_id')
            revision = adapter_info.metadata.get('revision', 'main')
            use_auth_token = adapter_info.metadata.get('auth_token')
            
            logger.info(f"Downloading adapter {adapter_id} from Hugging Face {repo_id}")
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=adapter_info.path,
                use_auth=use_auth_token is not None,
                token=use_auth_token
            )
            
        elif adapter_info.source == "aws":
            bucket = adapter_info.metadata.get('bucket')
            prefix = adapter_info.metadata.get('prefix')
            
            logger.info(f"Downloading adapter {adapter_id} from AWS S3 {bucket}/{prefix}")
            
            # List all objects with the given prefix
            response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
            
            # Download each file
            for obj in response.get('Contents', []):
                key = obj['Key']
                local_path = os.path.join(adapter_info.path, os.path.basename(key))
                self.s3_client.download_file(bucket, key, local_path)
            
        elif adapter_info.source == "gcp":
            bucket_name = adapter_info.metadata.get('bucket')
            prefix = adapter_info.metadata.get('prefix')
            
            logger.info(f"Downloading adapter {adapter_id} from GCP Storage {bucket_name}/{prefix}")
            
            bucket = self.gcs_client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            
            # Download each file
            for blob in blobs:
                local_path = os.path.join(adapter_info.path, os.path.basename(blob.name))
                blob.download_to_filename(local_path)
        
        return adapter_info.path
    
    def _make_space_if_needed(self):
        """
        Make space for a new adapter if maximum number of adapters in memory is reached
        by unloading the least recently used adapter
        """
        loaded_adapters = [adapter_id for adapter_id, info in self.adapters.items() if info.is_loaded]
        
        if len(loaded_adapters) >= self.max_adapters_in_memory:
            # Find the least recently used adapter
            least_recent_id = min(
                loaded_adapters, 
                key=lambda aid: self.adapters[aid].last_used
            )
            
            # Skip unloading if it's the current adapter
            if least_recent_id == self.current_adapter_id:
                # Find the second least recently used
                if len(loaded_adapters) > 1:
                    loaded_adapters.remove(least_recent_id)
                    least_recent_id = min(
                        loaded_adapters, 
                        key=lambda aid: self.adapters[aid].last_used
                    )
                else:
                    # Only one adapter loaded, and it's the current one
                    return
            
            # Unload the adapter
            logger.info(f"Unloading least recently used adapter: {least_recent_id}")
            self.adapters[least_recent_id].is_loaded = False
    
    def load_adapter(self, adapter_id: str) -> PeftModel:
        """
        Load an adapter onto the base model
        
        Args:
            adapter_id: Identifier for the adapter
            
        Returns:
            PeftModel: The model with the adapter applied
        """
        from peft import PeftModel
        
        if adapter_id not in self.adapters:
            raise ValueError(f"Adapter {adapter_id} not registered")
        
        # Check if already loaded as current adapter
        if adapter_id == self.current_adapter_id and self.current_adapter_model is not None:
            logger.info(f"Adapter {adapter_id} already loaded and active")
            # Update last used time
            self.adapters[adapter_id].last_used = time.time()
            return self.current_adapter_model
        
        # Download adapter if needed
        adapter_path = self._download_adapter_if_needed(adapter_id)
        
        # Make space for new adapter if needed
        self._make_space_if_needed()
        
        # Load the adapter
        logger.info(f"Loading adapter {adapter_id} from {adapter_path}")
        adapter_model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Update adapter info
        self.adapters[adapter_id].last_used = time.time()
        self.adapters[adapter_id].is_loaded = True
        
        # Update current adapter
        self.current_adapter_id = adapter_id
        self.current_adapter_model = adapter_model
        
        return adapter_model
    
    def get_current_adapter(self) -> Optional[tuple[str, PeftModel]]:
        """
        Get the currently active adapter
        
        Returns:
            Optional[tuple[str, PeftModel]]: Tuple of (adapter_id, adapter_model) or None if no adapter is loaded
        """
        if self.current_adapter_id is None or self.current_adapter_model is None:
            return None
        
        return (self.current_adapter_id, self.current_adapter_model)
    
    def get_adapter_info(self, adapter_id: str) -> AdapterInfo:
        """
        Get information about a registered adapter
        
        Args:
            adapter_id: Identifier for the adapter
            
        Returns:
            AdapterInfo: Information about the adapter
        """
        if adapter_id not in self.adapters:
            raise ValueError(f"Adapter {adapter_id} not registered")
        
        return self.adapters[adapter_id]
    
    def list_adapters(self) -> List[AdapterInfo]:
        """
        List all registered adapters
        
        Returns:
            List[AdapterInfo]: List of adapter information
        """
        return list(self.adapters.values())
    
    def unregister_adapter(self, adapter_id: str) -> bool:
        """
        Unregister an adapter
        
        Args:
            adapter_id: Identifier for the adapter
            
        Returns:
            bool: True if adapter was unregistered, False otherwise
        """
        if adapter_id not in self.adapters:
            return False
        
        # If it's the current adapter, unset it
        if adapter_id == self.current_adapter_id:
            self.current_adapter_id = None
            self.current_adapter_model = None
        
        # Remove adapter info
        del self.adapters[adapter_id]
        
        return True 