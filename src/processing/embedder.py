import numpy as np
from google import genai
from google.genai import types
from typing import List
from src.utils.config import Config

class Embedder:
    def __init__(self):
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        self.model = Config.EMBEDDING_MODEL

    def get_embedding(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        if not text.strip():
            text = "empty"
        
        response = self.client.models.embed_content(
            model=self.model,
            contents=[text],
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=1536
            )
        )
        
        vec = np.array(response.embeddings[0].values)
        
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
            
        return vec.tolist()