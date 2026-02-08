"""
Upsert synthetic entities to Pinecone with hybrid embeddings.

Processes synthetic entity files, generates dense and sparse embeddings,
fits BM25 models, and upserts to Pinecone namespaces.
"""

import glob
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.vector_db import VectorDB
from src.processing.data_processor import DataProcessor
from src.processing.embedder import Embedder
from src.utils.config import Config


class UpsertError(Exception):
    """Raised when upsert operation fails."""
    pass


def run_upsert(file_path: str):
    """
    Process and upsert entities from file to Pinecone.
    
    Args:
        file_path: Path to synthetic entities JSON file
        
    Raises:
        UpsertError: If upsert operation fails critically
    """
    processor = DataProcessor()
    vdb = VectorDB()

    entities = processor.load_json(file_path)
    if not entities:
        raise UpsertError(f"No entities loaded from {file_path}")

    base_name = os.path.basename(file_path)
    namespace = base_name.replace("synthetic_entities_", "").replace(".json", "")
    
    print(f"\n{'=' * 60}")
    print(f"Processing: {base_name}")
    print(f"Namespace: {namespace}")
    print(f"Entities: {len(entities)}")
    print(f"{'=' * 60}\n")

    # Initialize embedder without pre-fitted BM25
    embedder = Embedder(namespace=None)

    # Fit BM25 on document corpus
    print("Fitting BM25 encoder on document corpus...")
    corpus = [
        f"{e.get('name', '')} {e.get('type_name', '')} {e.get('description', '')}"
        for e in entities
    ]
    embedder.bm25.fit(corpus)
    print(f"BM25 fitted on {len(corpus)} documents")
    
    # Save BM25 model for later use during evaluation
    embedder.save_bm25(namespace)

    batch_size = Config.BATCH_SIZE
    total_upserted = 0

    for i in range(0, len(entities), batch_size):
        batch = entities[i:i + batch_size]
        vectors = []
        
        for entity in tqdm(
            batch,
            desc=f"Embedding Batch {i // batch_size + 1}",
            leave=False
        ):
            try:
                name = entity.get("name", "")
                type_name = entity.get("type_name", "")
                desc = entity.get("description", "")
                text = f"{name} {type_name} {desc}".strip()[:8000]
                
                dense_values = embedder.get_embedding(text)
                sparse_values = embedder.bm25.encode_documents(text)
                
                vectors.append({
                    "id": str(entity.get("guid", name if name else "unknown")),
                    "values": dense_values,
                    "sparse_values": sparse_values,
                    "metadata": {
                        "name": name,
                        "type_name": type_name,
                        "description": desc[:500],
                        "is_synthetic": entity.get("is_synthetic", False)
                    }
                })
            except Exception as ex:
                print(f"Embedding failed for {entity.get('name')}: {ex}")

        if vectors:
            result = vdb.upsert_vectors(vectors, namespace=namespace)
            if result:
                upserted = result.get("upserted_count", 0)
                total_upserted += upserted
                print(f"Batch {i // batch_size + 1}: Upserted {upserted} vectors")
            else:
                print(f"Warning: Upsert failed for batch {i // batch_size + 1}")
        
        time.sleep(2.0)

    print(f"\n{'=' * 60}")
    print(f"Namespace '{namespace}' finished!")
    print(f"Total upserted: {total_upserted}/{len(entities)}")
    print(f"{'=' * 60}\n")


def main():
    """Main entry point for upserting all synthetic entity files."""
    target_dir = project_root / Config.SYNTHETIC_DATA_DIR
    pattern = str(target_dir / "synthetic_entities_*.json")
    synthetic_files = glob.glob(pattern)
    
    if not synthetic_files:
        print(f"No synthetic entity files found in {target_dir}")
        return
    
    print(f"\nFound {len(synthetic_files)} synthetic entity files\n")
    
    successful = 0
    failed = 0
    
    for file_path in sorted(synthetic_files):
        try:
            run_upsert(file_path)
            successful += 1
        except UpsertError as e:
            print(f"Upsert error for {file_path}: {e}\n")
            failed += 1
        except Exception as e:
            print(f"Unexpected error processing {file_path}: {e}\n")
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Processing complete!")
    print(f"Successful: {successful}/{len(synthetic_files)}")
    print(f"Failed: {failed}/{len(synthetic_files)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()