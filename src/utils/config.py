"""Configuration settings for the RAG benchmarking system."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Centralized configuration for the RAG evaluation suite."""
    
    # API Keys
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    # Pinecone settings
    INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "wa-test")
    NAMESPACE = ""
    
    # Model settings
    EMBEDDING_MODEL = "models/gemini-embedding-001"
    EMBEDDING_DIM = 1536
    
    # Parameters for scripts
    BATCH_SIZE = 100
    TOP_K = 10
    ALPHA = 0.5  # 1.0 = Pure Dense, 0.0 = Pure Sparse
    
    # Directory paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
    QUERIES_DATA_DIR = DATA_DIR / "queries"
    RESULTS_DIR = BASE_DIR / "results"