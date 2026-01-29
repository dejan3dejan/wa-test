import time
import os
from tqdm import tqdm
from src.utils.config import Config
from src.processing.embedder import Embedder
from src.database.vector_db import VectorDB
from src.processing.data_processor import DataProcessor

def run_upsert(input_file):
    """Processes and uploads entities to Pinecone."""
    embedder = Embedder()
    vdb = VectorDB()
    processor = DataProcessor()

    # Load entities
    entities = processor.load_json(input_file)
    print(f"Upserting {len(entities)} entities from {input_file}")

    # Batch processing
    batch_size = Config.BATCH_SIZE
    for i in range(0, len(entities), batch_size):
        batch = entities[i : i + batch_size]
        vectors = []
        
        for e in tqdm(batch, desc=f"Batch {i//batch_size + 1}", leave=False):
            try:
                # Prepare text for embedding
                name = e.get('name', '')
                type_name = e.get('type_name', '')
                desc = e.get('description', '')
                
                text = f"{name} {type_name} {desc}".strip()[:8000]
                
                vectors.append({
                    "id": str(e.get("guid", name if name else "unknown")),
                    "values": embedder.get_embedding(text),
                    "metadata": {
                        "name": name,
                        "type_name": type_name,
                        "description": desc[:500],
                        "is_synthetic": e.get("is_synthetic", False)
                    }
                })
            except Exception as ex:
                print(f"Skipping {e.get('name')}: {ex}")

        if vectors:
            vdb.upsert_vectors(vectors)
        
        # Small delay to respect API rate limits
        time.sleep(1)

    print("Upsert completed successfully!")

if __name__ == "__main__":
    # Default input file
    INPUT_PATH = os.path.join(Config.SYNTHETIC_DATA_DIR, "synthetic_entities_10_mixed.json")
    run_upsert(INPUT_PATH)