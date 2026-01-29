import os
from dotenv import load_dotenv

load_dotenv()

class Config:
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

    DATA_DIR = "data"
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    SYNTHETIC_DATA_DIR = os.path.join(DATA_DIR, "synthetic")
    QUERIES_DATA_DIR = os.path.join(DATA_DIR, "queries")
    RESULTS_DIR = "results"