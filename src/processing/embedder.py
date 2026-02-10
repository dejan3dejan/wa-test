"""Embedder for dense and sparse vector generation."""

import os
import pickle
from typing import Dict, List

import numpy as np
import google.genai as genai
from google.genai import types
from pinecone_text.sparse import BM25Encoder

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbedderError(Exception):
    """Raised when embedding operations fail."""
    pass


class Embedder:
    """
    Handles dense (semantic) and sparse (keyword) embeddings for hybrid search.
    """
    
    def __init__(self, namespace: str = None):
        """
        Initialize embedder with Gemini client and BM25 encoder.
        
        Args:
            namespace: Optional namespace for loading pre-fitted BM25 model
        """
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        self.model = Config.EMBEDDING_MODEL
        self.namespace = namespace
        
        # Initialize BM25 encoder for sparse embeddings
        self.bm25 = BM25Encoder()
        
        # Auto-load BM25 if namespace provided and model exists
        if namespace:
            self.load_bm25(namespace)

    def get_embedding(
        self,
        text: str,
        task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> List[float]:
        """
        Generate dense embedding using Gemini.
        
        Args:
            text: Text to embed
            task_type: Task type for embedding (RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY)
            
        Returns:
            L2-normalized embedding vector
            
        Raises:
            EmbedderError: If embedding generation fails
        """
        if not text.strip():
            text = "empty"
        
        try:
            response = self.client.models.embed_content(
                model=self.model,
                contents=[text],
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=1536
                )
            )
            
            vec = np.array(response.embeddings[0].values)
            
            # L2 Normalization
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            else:
                raise EmbedderError("Embedding vector has zero norm")
                
            return vec.tolist()
            
        except Exception as e:
            raise EmbedderError(f"Failed to generate embedding: {e}") from e
    
    def get_sparse_embedding(self, text: str) -> Dict:
        """
        Generate sparse BM25 embedding for keyword matching.
        
        Args:
            text: Text to encode
            
        Returns:
            Sparse vector dictionary with indices and values
        """
        if not text.strip():
            text = "empty"
        return self.bm25.encode_queries(text)
    
    def save_bm25(self, namespace: str):
        """
        Save fitted BM25 model to disk for a specific namespace.
        
        Args:
            namespace: Namespace identifier for the model
            
        Raises:
            EmbedderError: If save operation fails
        """
        bm25_dir = Config.DATA_DIR / "bm25_models"
        bm25_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = bm25_dir / f"bm25_{namespace}.pkl"
        
        try:
            with open(model_path, "wb") as f:
                pickle.dump(self.bm25, f)
            logger.info(f"BM25 model saved: {model_path}")
        except (IOError, pickle.PicklingError) as e:
            raise EmbedderError(f"Error saving BM25 model: {e}") from e
    
    def load_bm25(self, namespace: str) -> bool:
        """
        Load fitted BM25 model from disk for a specific namespace.
        
        Args:
            namespace: Namespace identifier for the model
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        model_path = Config.DATA_DIR / "bm25_models" / f"bm25_{namespace}.pkl"
        
        if not model_path.exists():
            logger.warning(f"BM25 model not found: {model_path}")
            return False
        
        try:
            with open(model_path, "rb") as f:
                self.bm25 = pickle.load(f)
            logger.info(f"BM25 model loaded: {model_path}")
            return True
        except (IOError, pickle.UnpicklingError) as e:
            logger.error(f"Error loading BM25 model: {e}")
            return False