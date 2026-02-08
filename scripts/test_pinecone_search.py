import sys
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

from src.processing.embedder import Embedder
from src.database.vector_db import VectorDB

def run_search_experiment(input_file: str):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: Pinecone Search (Standard)")
    print(f"Input: {input_file}")
    print(f"{'='*60}")

    # 1. Load Input Data (Audited)
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter for FACTUAL only
    factual_entities = [e for e in data if e.get('audit_label') == 'factual']
    
    if not factual_entities:
        print("❌ No FACTUAL entities found! Did you audit the file?")
        return

    print(f"Loaded {len(data)} total. Using {len(factual_entities)} FACTUAL entities for testing.")

    # 2. Initialize System
    NAMESPACE = "10_clean" 
    embedder = Embedder(namespace=NAMESPACE)
    vdb = VectorDB()

    # 3. Run Search Loop
    correct_hits = 0
    
    for i, entity in enumerate(factual_entities):
        name = entity.get('name')
        description = entity.get('description')
        expected_id = entity.get('guid', name)
        
        print(f"[{i+1}/{len(factual_entities)}] Searching: {name}...")
        
        # A. Embed
        dense_vec = embedder.get_embedding(description, task_type="RETRIEVAL_QUERY")
        sparse_vec = embedder.get_sparse_embedding(description)

        # B. Query Pinecone
        start_t = time.time()
        results = vdb.query_index(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=5,
            namespace=NAMESPACE 
        )
        latency = (time.time() - start_t) * 1000
    
        # C. Check Result
        found = False
        if results and results.get('matches'):
            top_match = results['matches'][0]
            match_name = top_match.get('metadata', {}).get('name', '')
            if name.lower() in match_name.lower():
                found = True
        
        if found:
            correct_hits += 1
            print(f"   ✅ Found (Latency: {latency:.2f}ms)")
        else:
            print(f"   ❌ Not Found (Latency: {latency:.2f}ms)")

    print(f"\n{'-'*60}")
    print(f"Experiment Complete.")
    print(f"Accuracy: {correct_hits}/{len(factual_entities)} ({(correct_hits/len(factual_entities)):.2%})")
    print(f"{'-'*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_pinecone_search.py <path_to_audited_json>")
    else:
        run_search_experiment(sys.argv[1])
