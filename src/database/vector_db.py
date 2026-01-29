from pinecone import Pinecone
from typing import List, Dict
from src.utils.config import Config

class VectorDB:
    def __init__(self):
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index = self.pc.Index(Config.INDEX_NAME)

    def upsert_vectors(self, vectors: List[Dict]):
        """Upserts vectors into Pinecone."""
        try:
            return self.index.upsert(vectors=vectors)
        except Exception as e:
            print(f"Error during upsert: {e}")
            return None

    def query_index(self, vector: List[float], top_k: int = Config.TOP_K) -> Dict:
        """Searches the index."""
        try:
            return self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True
            )
        except Exception as e:
            print(f"Error during search: {e}")
            return {}

    def get_stats(self):
        """Returns index statistics."""
        return self.index.describe_index_stats()