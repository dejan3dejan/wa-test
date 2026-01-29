import pandas as pd
from tqdm import tqdm
from src.utils.config import Config
from src.processing.embedder import Embedder
from src.database.vector_db import VectorDB

class Evaluator:
    def __init__(self):
        self.embedder = Embedder()
        self.vdb = VectorDB()

    def run(self, queries: list) -> pd.DataFrame:
        results = []
        for q in tqdm(queries, desc="Testing"):
            query_text = q["query"]
            expected_id = str(q["expected_object_id"])
            
            # 1. Getting embeddings
            vec = self.embedder.get_embedding(query_text, task_type="RETRIEVAL_QUERY")
            
            # 2. Searching in Pinecone
            resp = self.vdb.query_index(vector=vec, top_k=Config.TOP_K)
            
            matches = resp.get('matches', [])
            ids = [str(m['id']) for m in matches]
            scores = [m['score'] for m in matches]
            names = [m.get('metadata', {}).get('name', 'N/A') for m in matches]
            
            try:
                rank = ids.index(expected_id) + 1
                score = scores[rank - 1]
            except ValueError:
                rank = 999
                score = 0.0

            results.append({
                "query": query_text,
                "type": q.get("query_type", "unknown"),
                "difficulty": q.get("difficulty", "unknown"),
                "expected_id": expected_id,
                "expected_name": q.get("expected_object_name", ""),
                "top1_id": ids[0] if ids else None,
                "top1_name": names[0] if names else None,
                "top1_score": scores[0] if scores else 0.0,
                "rank": rank,
                "hit@1": 1 if rank == 1 else 0,
                "hit@3": 1 if rank <= 3 else 0,
                "hit@5": 1 if rank <= 5 else 0,
                "hit@10": 1 if rank <= 10 else 0,
                "mrr": 1.0 / rank if rank < 999 else 0.0
            })
        return pd.DataFrame(results)