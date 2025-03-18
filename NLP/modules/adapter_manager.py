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
        Initialize the MultiAdapterManager.
        
        Args:
            base_model: The base model to which adapters will be applied.
            device: The device to use for model operations (default is CUDA if available).
            cache_dir: Directory to cache adapters.
            max_adapters_in_memory: Maximum number of adapters to keep in memory.
            aws_credentials: AWS credentials for accessing S3.
            gcp_credentials: Path to GCP credentials file.
        """
        self.base_model = base_model
        self.device = device
        self.cache_dir = cache_dir or tempfile.mkdtemp()
        self.max_adapters_in_memory = max_adapters_in_memory
        self.aws_credentials = aws_credentials
        self.gcp_credentials = gcp_credentials
        self.adapters = {}  # Dictionary to store adapter information
        self._setup_cloud_clients()
    
    def _setup_cloud_clients(self):
        """Set up clients for cloud storage services (AWS and GCP)."""
        # Initialize AWS S3 client if credentials are provided
        if self.aws_credentials:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_credentials.get('aws_access_key_id'),
                aws_secret_access_key=self.aws_credentials.get('aws_secret_access_key')
            )
        else:
            self.s3_client = None

        # Initialize GCP Storage client if credentials are provided
        if self.gcp_credentials:
            self.gcp_client = storage.Client.from_service_account_json(self.gcp_credentials)
        else:
            self.gcp_client = None
    
    def register_adapter_from_local(
        self, 
        adapter_id: str,
        adapter_path: str,
        adapter_name: Optional[str] = None,
        adapter_type: str = "lora",
        metadata: Optional[Dict[str, Any]] = None
    ) -> AdapterInfo:
        """
        Register an adapter from a local path.
        
        Args:
            adapter_id: Unique identifier for the adapter.
            adapter_path: Path to the adapter files.
            adapter_name: Optional name for the adapter.
            adapter_type: Type of adapter (e.g., 'lora').
            metadata: Additional metadata for the adapter.
        
        Returns:
            AdapterInfo: Information about the registered adapter.
        """
        # Check if the adapter path exists
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

        # Create AdapterInfo object
        adapter_info = AdapterInfo(
            adapter_id=adapter_id,
            adapter_name=adapter_name or adapter_id,
            adapter_type=adapter_type,
            source='local',
            path=adapter_path,
            metadata=metadata or {},
            last_used=time.time()
        )

        # Store adapter information
        self.adapters[adapter_id] = adapter_info
        logger.info(f"Registered local adapter: {adapter_id}")
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
        Register an adapter from the Hugging Face Hub.
        
        Args:
            adapter_id: Unique identifier for the adapter.
            repo_id: Repository ID on Hugging Face Hub.
            adapter_name: Optional name for the adapter.
            adapter_type: Type of adapter (e.g., 'lora').
            metadata: Additional metadata for the adapter.
            revision: Revision of the repository to use.
            use_auth_token: Authentication token for private repositories.
        
        Returns:
            AdapterInfo: Information about the registered adapter.
        """
        # Download adapter from Hugging Face Hub
        adapter_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            use_auth_token=use_auth_token,
            cache_dir=self.cache_dir
        )

        # Create AdapterInfo object
        adapter_info = AdapterInfo(
            adapter_id=adapter_id,
            adapter_name=adapter_name or adapter_id,
            adapter_type=adapter_type,
            source='huggingface',
            path=adapter_path,
            metadata=metadata or {},
            last_used=time.time()
        )

        # Store adapter information
        self.adapters[adapter_id] = adapter_info
        logger.info(f"Registered Hugging Face adapter: {adapter_id}")
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
        Register an adapter from AWS S3.
        
        Args:
            adapter_id: Unique identifier for the adapter.
            bucket: S3 bucket name.
            prefix: Prefix path in the S3 bucket.
            adapter_name: Optional name for the adapter.
            adapter_type: Type of adapter (e.g., 'lora').
            metadata: Additional metadata for the adapter.
        
        Returns:
            AdapterInfo: Information about the registered adapter.
        """
        # Download adapter from S3
        adapter_path = os.path.join(self.cache_dir, adapter_id)
        os.makedirs(adapter_path, exist_ok=True)

        # List objects in the specified S3 prefix
        response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for obj in response.get('Contents', []):
            key = obj['Key']
            file_name = os.path.basename(key)
            local_file_path = os.path.join(adapter_path, file_name)
            self.s3_client.download_file(bucket, key, local_file_path)

        # Create AdapterInfo object
        adapter_info = AdapterInfo(
            adapter_id=adapter_id,
            adapter_name=adapter_name or adapter_id,
            adapter_type=adapter_type,
            source='aws',
            path=adapter_path,
            metadata=metadata or {},
            last_used=time.time()
        )

        # Store adapter information
        self.adapters[adapter_id] = adapter_info
        logger.info(f"Registered AWS adapter: {adapter_id}")
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
        Register an adapter from GCP Storage.
        
        Args:
            adapter_id: Unique identifier for the adapter.
            bucket: GCP bucket name.
            prefix: Prefix path in the GCP bucket.
            adapter_name: Optional name for the adapter.
            adapter_type: Type of adapter (e.g., 'lora').
            metadata: Additional metadata for the adapter.
        
        Returns:
            AdapterInfo: Information about the registered adapter.
        """
        # Download adapter from GCP Storage
        adapter_path = os.path.join(self.cache_dir, adapter_id)
        os.makedirs(adapter_path, exist_ok=True)

        # List blobs in the specified GCP prefix
        bucket = self.gcp_client.bucket(bucket)
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            file_name = os.path.basename(blob.name)
            local_file_path = os.path.join(adapter_path, file_name)
            blob.download_to_filename(local_file_path)

        # Create AdapterInfo object
        adapter_info = AdapterInfo(
            adapter_id=adapter_id,
            adapter_name=adapter_name or adapter_id,
            adapter_type=adapter_type,
            source='gcp',
            path=adapter_path,
            metadata=metadata or {},
            last_used=time.time()
        )

        # Store adapter information
        self.adapters[adapter_id] = adapter_info
        logger.info(f"Registered GCP adapter: {adapter_id}")
        return adapter_info
    
    def _download_adapter_if_needed(self, adapter_id: str) -> str:
        """
        Download the adapter if it is not already cached.
        
        Args:
            adapter_id: Unique identifier for the adapter.
        
        Returns:
            str: Path to the downloaded adapter.
        """
        adapter_info = self.adapters.get(adapter_id)
        if not adapter_info:
            raise ValueError(f"Adapter not registered: {adapter_id}")

        # Check if the adapter is already downloaded
        if os.path.exists(adapter_info.path):
            return adapter_info.path

        # Download logic based on source
        if adapter_info.source == 'huggingface':
            # Re-download from Hugging Face Hub
            adapter_info.path = snapshot_download(
                repo_id=adapter_info.metadata['repo_id'],
                revision=adapter_info.metadata.get('revision', 'main'),
                use_auth_token=adapter_info.metadata.get('use_auth_token'),
                cache_dir=self.cache_dir
            )
        elif adapter_info.source == 'aws':
            # Re-download from AWS S3
            self.register_adapter_from_aws(
                adapter_id=adapter_id,
                bucket=adapter_info.metadata['bucket'],
                prefix=adapter_info.metadata['prefix'],
                adapter_name=adapter_info.adapter_name,
                adapter_type=adapter_info.adapter_type,
                metadata=adapter_info.metadata
            )
        elif adapter_info.source == 'gcp':
            # Re-download from GCP Storage
            self.register_adapter_from_gcp(
                adapter_id=adapter_id,
                bucket=adapter_info.metadata['bucket'],
                prefix=adapter_info.metadata['prefix'],
                adapter_name=adapter_info.adapter_name,
                adapter_type=adapter_info.adapter_type,
                metadata=adapter_info.metadata
            )

        return adapter_info.path
    
    def _make_space_if_needed(self):
        """
        Ensure there is enough space in memory for new adapters by unloading the least recently used adapter.
        """
        if len(self.adapters) > self.max_adapters_in_memory:
            # Sort adapters by last used time
            sorted_adapters = sorted(self.adapters.values(), key=lambda x: x.last_used)
            # Unload the least recently used adapter
            lru_adapter = sorted_adapters[0]
            self.adapters.pop(lru_adapter.adapter_id)
            logger.info(f"Unloaded adapter: {lru_adapter.adapter_id}")
    
    def load_adapter(self, adapter_id: str) -> PeftModel:
        """
        Load an adapter onto the base model.
        
        Args:
            adapter_id: Unique identifier for the adapter.
        
        Returns:
            PeftModel: The model with the adapter applied.
        """
        adapter_info = self.adapters.get(adapter_id)
        if not adapter_info:
            raise ValueError(f"Adapter not registered: {adapter_id}")

        # Ensure the adapter is downloaded
        adapter_path = self._download_adapter_if_needed(adapter_id)

        # Load the adapter onto the base model
        adapter_model = PeftModel.from_pretrained(
            self.base_model,
            adapter_path,
            device_map="auto" if self.device == "cuda" else None
        )
        adapter_info.is_loaded = True
        adapter_info.last_used = time.time()
        logger.info(f"Loaded adapter: {adapter_id}")
        return adapter_model
    
    def get_current_adapter(self) -> Optional[tuple[str, PeftModel]]:
        """
        Get the currently loaded adapter and its model.
        
        Returns:
            Optional[tuple[str, PeftModel]]: Tuple of adapter ID and model, or None if no adapter is loaded.
        """
        for adapter_id, adapter_info in self.adapters.items():
            if adapter_info.is_loaded:
                return adapter_id, PeftModel.from_pretrained(
                    self.base_model,
                    adapter_info.path,
                    device_map="auto" if self.device == "cuda" else None
                )
        return None
    
    def get_adapter_info(self, adapter_id: str) -> AdapterInfo:
        """
        Get information about a registered adapter.
        
        Args:
            adapter_id: Unique identifier for the adapter.
        
        Returns:
            AdapterInfo: Information about the adapter.
        """
        adapter_info = self.adapters.get(adapter_id)
        if not adapter_info:
            raise ValueError(f"Adapter not registered: {adapter_id}")
        return adapter_info
    
    def list_adapters(self) -> List[AdapterInfo]:
        """
        List all registered adapters.
        
        Returns:
            List[AdapterInfo]: List of adapter information.
        """
        return list(self.adapters.values())
    
    def unregister_adapter(self, adapter_id: str) -> bool:
        """
        Unregister an adapter, removing it from memory.
        
        Args:
            adapter_id: Unique identifier for the adapter.
        
        Returns:
            bool: True if the adapter was successfully unregistered, False otherwise.
        """
        if adapter_id in self.adapters:
            del self.adapters[adapter_id]
            logger.info(f"Unregistered adapter: {adapter_id}")
            return True
        return False 