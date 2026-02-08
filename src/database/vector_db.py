"""Pinecone vector database interface for hybrid search operations."""

from typing import Dict, List, Optional

from pinecone import Pinecone

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorDBError(Exception):
    """Raised when vector database operations fail."""
    pass


class VectorDB:
    """Interface for Pinecone vector database operations."""
    
    def __init__(self, api_key: Optional[str] = None, index_name: Optional[str] = None):
        """
        Initialize Pinecone client and index.
        
        Args:
            api_key: Pinecone API key. Defaults to Config.PINECONE_API_KEY
            index_name: Index name. Defaults to Config.INDEX_NAME
            
        Raises:
            VectorDBError: If initialization fails
        """
        try:
            self.pc = Pinecone(api_key=api_key or Config.PINECONE_API_KEY)
            self.index = self.pc.Index(index_name or Config.INDEX_NAME)
        except Exception as e:
            raise VectorDBError(f"Failed to initialize VectorDB: {e}") from e

    def upsert_vectors(
        self,
        vectors: List[Dict],
        namespace: str = Config.NAMESPACE
    ) -> Optional[Dict]:
        """
        Upsert a batch of vectors to Pinecone.
        
        Args:
            vectors: List of vector dicts with 'id', 'values', 
                     'sparse_values', and 'metadata'
            namespace: Partition namespace (e.g., 'production', 'benchmark')
            
        Returns:
            Upsert response dict or None if failed
        """
        if not vectors:
            logger.warning("Attempted to upsert empty vector list")
            return None
        
        try:
            response = self.index.upsert(vectors=vectors, namespace=namespace)
            logger.debug(
                f"Upserted {len(vectors)} vectors to namespace '{namespace}'"
            )
            return response
            
        except Exception as e:
            logger.error(f"Error during upsert to namespace '{namespace}': {e}")
            return None

    def query_index(
        self,
        vector: list,
        sparse_vector: Optional[dict] = None,
        top_k: int = Config.TOP_K,
        namespace: str = Config.NAMESPACE
    ) -> Dict:
        """
        Query the Pinecone index with hybrid search (dense + sparse).
        
        Args:
            vector: Dense embedding list
            sparse_vector: Sparse vector dict with 'indices' and 'values'
            top_k: Number of results to return
            namespace: Partition namespace to search
            
        Returns:
            Query response dict with matches, or empty dict if failed
        """
        if not vector:
            logger.error("Attempted to query with empty vector")
            return {}
        
        try:
            response = self.index.query(
                vector=vector,
                sparse_vector=sparse_vector,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace
            )
            return response
            
        except Exception as e:
            logger.error(
                f"Error during query in namespace '{namespace}': {e}"
            )
            return {}

    def get_stats(self) -> Dict:
        """
        Get index statistics.
        
        Returns:
            Index statistics dict
            
        Raises:
            VectorDBError: If stats retrieval fails
        """
        try:
            return self.index.describe_index_stats()
        except Exception as e:
            raise VectorDBError(f"Failed to get index stats: {e}") from e
    
    def delete_namespace(self, namespace: str, delete_all: bool = False):
        """
        Delete all vectors in a namespace.
        
        Args:
            namespace: Namespace to delete
            delete_all: If True, confirms deletion of all vectors
            
        Raises:
            VectorDBError: If deletion fails
        """
        if not delete_all:
            logger.warning(
                f"Skipping deletion of namespace '{namespace}' "
                f"(delete_all=False)"
            )
            return
        
        try:
            self.index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Deleted all vectors in namespace '{namespace}'")
        except Exception as e:
            raise VectorDBError(
                f"Failed to delete namespace '{namespace}': {e}"
            ) from e