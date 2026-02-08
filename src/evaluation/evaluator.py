"""Enterprise-grade retrieval evaluation for RAG benchmarking."""

from typing import Optional

import pandas as pd
from tqdm import tqdm

from src.database.vector_db import VectorDB
from src.processing.embedder import Embedder
from src.utils.config import Config


class Evaluator:
    """
    Enterprise-grade evaluator for RAG retrieval performance.
    
    Supports dependency injection for easier testing and mocking.
    """
    
    def __init__(
        self,
        vdb: Optional[VectorDB] = None,
        embedder: Optional[Embedder] = None
    ):
        """
        Initialize evaluator with optional dependencies.
        
        Args:
            vdb: Vector database instance. If None, creates new instance.
            embedder: Embedder instance. If None, creates new instance per namespace.
        """
        self.vdb = vdb if vdb is not None else VectorDB()
        self.embedder = embedder

    def run(
        self,
        queries: list,
        namespace: str = Config.NAMESPACE
    ) -> pd.DataFrame:
        """
        Evaluate retrieval performance on a list of queries.
        
        Args:
            queries: List of query dictionaries with expected results
            namespace: Pinecone namespace to query
            
        Returns:
            DataFrame with evaluation metrics per query
        """
        results = []
        
        # Initialize embedder for this namespace if not provided
        if self.embedder is None:
            self.embedder = Embedder(namespace=namespace)

        for query_dict in tqdm(queries, desc=f"Testing [{namespace}]", leave=False):
            query_text = query_dict.get("query", "")
            expected_id = str(query_dict.get("expected_object_id", ""))
            
            response = self._perform_search(query_text, namespace)
            
            # Extract matches and compute metrics
            matches = response.get("matches", [])
            retrieved_ids = [str(match["id"]) for match in matches]
            scores = [match["score"] for match in matches]
            
            rank, score = self._calculate_rank_and_score(
                expected_id, retrieved_ids, scores
            )

            results.append({
                "query": query_text,
                "type": query_dict.get("query_type", "unknown"),
                "difficulty": query_dict.get("difficulty", "unknown"),
                "expected_id": expected_id,
                "expected_name": query_dict.get("expected_object_name", ""),
                "top1_id": retrieved_ids[0] if retrieved_ids else None,
                "top1_name": self._extract_match_name(matches[0]) if matches else None,
                "top1_score": scores[0] if scores else 0.0,
                "rank": rank,
                "hit@1": 1 if rank == 1 else 0,
                "hit@3": 1 if rank <= 3 else 0,
                "hit@5": 1 if rank <= 5 else 0,
                "hit@10": 1 if rank <= 10 else 0,
                "mrr": 1.0 / rank if rank < 999 else 0.0
            })
                
        return pd.DataFrame(results)

    def _perform_search(self, query_text: str, namespace: str) -> dict:
        """
        Execute hybrid search using dense and sparse vectors.
        
        Args:
            query_text: Query string
            namespace: Pinecone namespace
            
        Returns:
            Search response from vector database
        """
        dense_vec = self.embedder.get_embedding(
            query_text, task_type="RETRIEVAL_QUERY"
        )
        sparse_vec = self.embedder.get_sparse_embedding(query_text)
        
        return self.vdb.query_index(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            namespace=namespace
        )
    
    def _calculate_rank_and_score(
        self,
        expected_id: str,
        retrieved_ids: list,
        scores: list
    ) -> tuple:
        """
        Calculate rank and score of expected result in retrieved results.
        
        Args:
            expected_id: Expected object ID
            retrieved_ids: List of retrieved IDs
            scores: Corresponding scores
            
        Returns:
            Tuple of (rank, score). Rank is 999 if not found.
        """
        try:
            rank = retrieved_ids.index(expected_id) + 1
            score = scores[rank - 1]
        except ValueError:
            rank = 999
            score = 0.0
        
        return rank, score
    
    @staticmethod
    def _extract_match_name(match: dict) -> str:
        """
        Extract name from match metadata.
        
        Args:
            match: Match dictionary from vector database
            
        Returns:
            Name from metadata or 'N/A'
        """
        return match.get("metadata", {}).get("name", "N/A")