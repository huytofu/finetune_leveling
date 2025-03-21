"""
Advanced NLP Monitoring Module.

This module provides advanced NLP-specific metrics and monitoring capabilities:
1. Perplexity and coherence scoring
2. Topic distribution monitoring
3. Semantic drift detection
4. Visualization utilities
"""

import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

from .error_handler import ErrorHandler

logger = logging.getLogger(__name__)

class AdvancedNLPMonitor:
    """
    Advanced NLP monitoring with specialized metrics and visualizations.
    """
    
    def __init__(
        self,
        perplexity_model: str = "gpt2",
        embedding_model: str = "all-MiniLM-L6-v2",
        n_topics: int = 10,
        visualization_dir: str = "monitoring_plots"
    ):
        self.n_topics = n_topics
        self.visualization_dir = Path(visualization_dir)
        self.visualization_dir.mkdir(exist_ok=True)
        
        # Initialize models
        logger.info("Initializing NLP models...")
        try:
            # Perplexity model
            self.perplexity_tokenizer = GPT2Tokenizer.from_pretrained(perplexity_model)
            self.perplexity_model = GPT2LMHeadModel.from_pretrained(perplexity_model)
            
            # Semantic similarity model
            self.embedding_model = SentenceTransformer(embedding_model)
            
            # Topic modeling
            self.vectorizer = CountVectorizer(max_features=5000, stop_words='english')
            self.lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42
            )
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            ErrorHandler.handle_error(e, "model_initialization")
            raise
            
    def compute_perplexity(self, texts: List[str], batch_size: int = 8) -> Dict[str, float]:
        """Compute perplexity scores for input texts."""
        try:
            perplexities = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                encodings = self.perplexity_tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                with torch.no_grad():
                    outputs = self.perplexity_model(**encodings)
                    loss = outputs.loss
                    perplexity = torch.exp(loss).item()
                    perplexities.append(perplexity)
                    
            avg_perplexity = np.mean(perplexities)
            std_perplexity = np.std(perplexities)
            
            return {
                "mean_perplexity": avg_perplexity,
                "std_perplexity": std_perplexity,
                "min_perplexity": min(perplexities),
                "max_perplexity": max(perplexities)
            }
            
        except Exception as e:
            logger.error(f"Error computing perplexity: {str(e)}")
            ErrorHandler.handle_error(e, "perplexity_computation")
            return {}
            
    def compute_coherence(self, texts: List[str]) -> Dict[str, float]:
        """Compute model coherence scores."""
        try:
            # Prepare documents
            processed_docs = [text.lower().split() for text in texts]
            dictionary = Dictionary(processed_docs)
            
            # Create corpus
            corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
            
            # Train LDA model
            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=self.n_topics,
                random_state=42
            )
            
            # Compute coherence scores
            c_v = CoherenceModel(
                model=lda_model,
                texts=processed_docs,
                dictionary=dictionary,
                coherence='c_v'
            ).get_coherence()
            
            u_mass = CoherenceModel(
                model=lda_model,
                corpus=corpus,
                dictionary=dictionary,
                coherence='u_mass'
            ).get_coherence()
            
            return {
                "c_v_coherence": c_v,
                "u_mass_coherence": u_mass
            }
            
        except Exception as e:
            logger.error(f"Error computing coherence: {str(e)}")
            ErrorHandler.handle_error(e, "coherence_computation")
            return {}
            
    def monitor_topic_distribution(
        self,
        reference_texts: List[str],
        current_texts: List[str]
    ) -> Dict[str, Any]:
        """Monitor changes in topic distribution."""
        try:
            # Fit vectorizer on reference texts
            ref_vectors = self.vectorizer.fit_transform(reference_texts)
            
            # Fit LDA on reference data
            self.lda.fit(ref_vectors)
            
            # Transform both datasets
            ref_topics = self.lda.transform(ref_vectors)
            current_vectors = self.vectorizer.transform(current_texts)
            current_topics = self.lda.transform(current_vectors)
            
            # Compute distribution differences
            topic_drift = jensenshannon(
                ref_topics.mean(axis=0),
                current_topics.mean(axis=0)
            )
            
            # Get top words per topic
            feature_names = self.vectorizer.get_feature_names_out()
            top_words = []
            for topic_idx, topic in enumerate(self.lda.components_):
                top_words_idx = topic.argsort()[:-10:-1]
                top_words.append([feature_names[i] for i in top_words_idx])
                
            # Visualize topic distributions
            self._plot_topic_distributions(
                ref_topics.mean(axis=0),
                current_topics.mean(axis=0),
                top_words
            )
            
            return {
                "topic_drift_score": float(topic_drift),
                "reference_distribution": ref_topics.mean(axis=0).tolist(),
                "current_distribution": current_topics.mean(axis=0).tolist(),
                "top_words_per_topic": top_words
            }
            
        except Exception as e:
            logger.error(f"Error monitoring topic distribution: {str(e)}")
            ErrorHandler.handle_error(e, "topic_monitoring")
            return {}
            
    def detect_semantic_drift(
        self,
        reference_texts: List[str],
        current_texts: List[str],
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Detect semantic drift using embedding space analysis."""
        try:
            # Compute embeddings
            ref_embeddings = self.embedding_model.encode(
                reference_texts,
                batch_size=batch_size,
                show_progress_bar=False
            )
            current_embeddings = self.embedding_model.encode(
                current_texts,
                batch_size=batch_size,
                show_progress_bar=False
            )
            
            # Compute centroid distances
            ref_centroid = np.mean(ref_embeddings, axis=0)
            current_centroid = np.mean(current_embeddings, axis=0)
            centroid_distance = np.linalg.norm(ref_centroid - current_centroid)
            
            # Compute distribution statistics
            ref_distances = np.linalg.norm(ref_embeddings - ref_centroid, axis=1)
            current_distances = np.linalg.norm(current_embeddings - current_centroid, axis=1)
            
            # Visualize semantic drift
            self._plot_semantic_drift(ref_distances, current_distances)
            
            return {
                "centroid_distance": float(centroid_distance),
                "reference_spread": float(np.std(ref_distances)),
                "current_spread": float(np.std(current_distances)),
                "semantic_drift_score": float(centroid_distance * 
                    (np.std(current_distances) / np.std(ref_distances)))
            }
            
        except Exception as e:
            logger.error(f"Error detecting semantic drift: {str(e)}")
            ErrorHandler.handle_error(e, "semantic_drift")
            return {}
            
    def _plot_topic_distributions(
        self,
        ref_dist: np.ndarray,
        current_dist: np.ndarray,
        top_words: List[List[str]]
    ):
        """Plot topic distribution comparison."""
        plt.figure(figsize=(15, 8))
        x = np.arange(len(ref_dist))
        width = 0.35
        
        plt.bar(x - width/2, ref_dist, width, label='Reference')
        plt.bar(x + width/2, current_dist, width, label='Current')
        
        plt.xlabel('Topics')
        plt.ylabel('Distribution')
        plt.title('Topic Distribution Comparison')
        plt.legend()
        
        # Add top words as x-tick labels
        plt.xticks(x, [f"Topic {i+1}" for i in range(len(ref_dist))], rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.visualization_dir / 'topic_distribution.png')
        plt.close()
        
    def _plot_semantic_drift(
        self,
        ref_distances: np.ndarray,
        current_distances: np.ndarray
    ):
        """Plot semantic drift visualization."""
        plt.figure(figsize=(10, 6))
        
        sns.kdeplot(data=ref_distances, label='Reference', fill=True)
        sns.kdeplot(data=current_distances, label='Current', fill=True)
        
        plt.xlabel('Distance from Centroid')
        plt.ylabel('Density')
        plt.title('Semantic Drift Analysis')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.visualization_dir / 'semantic_drift.png')
        plt.close() 