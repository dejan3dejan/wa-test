"""Main FastAPI application for RAG operations."""

import os
import json
import glob
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from src.api import schemas
from src.database.vector_db import VectorDB
from src.processing.embedder import Embedder
from src.utils.config import Config
from src.utils.logger import get_logger, Logger

Logger.setup(log_level="INFO", log_file="app.log")
logger = get_logger(__name__)
resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle app startup and shutdown events."""
    logger.info("Initializing API resources...")
    resources["vector_db"] = VectorDB()
    # Embedder needs a namespace to load BM25, but we can also load it dynamically
    resources["embedder"] = Embedder() 
    yield
    logger.info("Shutting down API resources...")
    resources.clear()

app = FastAPI(
    title="Industrial RAG API",
    description="API for hybrid search and auditing in an industrial LLM context.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENDPOINTS ---

@app.get("/health")
async def health_check():
    """Basic health check to verify the API is running."""
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/stats", response_model=schemas.StatsResponse)
async def get_stats():
    """Get statistics about the Pinecone index."""
    try:
        db: VectorDB = resources["vector_db"]
        stats = db.get_stats()
        return {
            "total_vector_count": stats.get("total_vector_count", 0),
            "namespaces": stats.get("namespaces", {}),
            "index_fullness": stats.get("index_fullness", 0.0)
        }
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=schemas.QueryResponse)
async def query_rag(request: schemas.QueryRequest):
    """
    Perform hybrid search and optionally audit the result.
    
    LEARN: 
    1. We use 'async def' because API calls to Gemini and Pinecone are I/O bound.
    2. 'await' allows the server to handle other requests while waiting for these calls.
    """
    try:
        db: VectorDB = resources["vector_db"]
        embedder: Embedder = resources["embedder"]

        if request.namespace:
            embedder.load_bm25(request.namespace)
        
        dense_vec = embedder.get_embedding(request.query, task_type="RETRIEVAL_QUERY")
        sparse_vec = embedder.get_sparse_embedding(request.query)
        
        response = db.query_index(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=request.top_k,
            namespace=request.namespace
        )
        
        results = []
        for match in response.get("matches", []):
            item = schemas.SearchResult(
                id=match["id"],
                score=match["score"],
                metadata=match["metadata"]
            )
            results.append(item)
            
        if request.with_audit and results:
            top_result = results[0]
            audit = await perform_audit(
                embedder, 
                query=request.query, 
                document=top_result.metadata.get("description", "")
            )
            top_result.audit = audit
            
        return schemas.QueryResponse(
            query=request.query,
            results=results,
            namespace=request.namespace
        )

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

async def perform_audit(embedder: Embedder, query: str, document: str) -> schemas.AuditResult:
    """Uses Gemini to judge if the retrieved document actually answers the query."""

    try:
        system_instruction = """You are a Data Auditor. Verify if the Description matches the Query.
        Label 'factual' if it matches, 'hallucinated' if it contradicts or is irrelevant."""
        
        response = embedder.client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=f"Query: {query}\nDocument: {document}\n\nAnalyze accuracy and return JSON.",
            config={
                "response_mime_type": "application/json",
                "response_schema": schemas.AuditResult,
                "system_instruction": system_instruction
            }
        )
        return schemas.AuditResult.model_validate_json(response.text)
    except Exception as e:
        logger.error(f"Audit failed: {e}")
        return schemas.AuditResult(
            label="hallucinated", 
            reasoning=f"Audit process error: {str(e)}", 
            confidence_score=0.0
        )

@app.post("/upsert")
async def upsert_data(request: schemas.UpsertRequest):
    """
    Ingest a batch of documents into Pinecone.
    
    LEARN: 
    1. We iterate through items, generate embeddings, and prepare them for Pinecone.
    2. In a large-scale system, you might want to use BackgroundTasks 
       to avoid keeping the user waiting while Gemini embeds all items.
    """
    try:
        db: VectorDB = resources["vector_db"]
        embedder: Embedder = resources["embedder"]
        
        vectors_to_upsert = []
        
        for item in request.items:
            # Generate vectors
            dense_vec = embedder.get_embedding(item.text, task_type="RETRIEVAL_DOCUMENT")
            sparse_vec = embedder.get_sparse_embedding(item.text)
            
            # Prepare Pinecone format
            vector = {
                "id": item.id,
                "values": dense_vec,
                "sparse_values": sparse_vec,
                "metadata": {**item.metadata, "description": item.text}
            }
            vectors_to_upsert.append(vector)
            
        # Upsert in batch
        response = db.upsert_vectors(vectors_to_upsert, namespace=request.namespace)
        
        return {
            "message": f"Successfully upserted {len(vectors_to_upsert)} items.",
            "namespace": request.namespace,
            "upsert_count": len(vectors_to_upsert)
        }

    except Exception as e:
        logger.error(f"Upsert error: {e}")
        raise HTTPException(status_code=500, detail=f"Upsert failed: {str(e)}")

@app.delete("/namespace/{name}")
async def delete_namespace(name: str):
    """Delete a specific namespace from the index."""
    try:
        db: VectorDB = resources["vector_db"]
        db.delete_namespace(namespace=name, delete_all=True)
        return {"message": f"Namespace '{name}' deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/multi-query")
async def multi_query_rag(request: schemas.MultiQueryRequest):
    """Perform multiple hybrid search queries in one request."""
    try:
        db: VectorDB = resources["vector_db"]
        embedder: Embedder = resources["embedder"]
        if request.namespace:
            embedder.load_bm25(request.namespace)
        
        all_results = []
        for query_text in request.queries:
            dense_vec = embedder.get_embedding(query_text, task_type="RETRIEVAL_QUERY")
            sparse_vec = embedder.get_sparse_embedding(query_text)
            response = db.query_index(vector=dense_vec, sparse_vector=sparse_vec, top_k=request.top_k, namespace=request.namespace)
            
            matches = [schemas.SearchResult(id=m["id"], score=m["score"], metadata=m["metadata"]) for m in response.get("matches", [])]
            all_results.append({"query": query_text, "results": matches})
            
        return {"results": all_results, "namespace": request.namespace}
    except Exception as e:
        logger.error(f"Multi-query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{doc_id}")
async def get_document(doc_id: str, namespace: str = ""):
    """Retrieve exactly one document by ID."""
    try:
        db: VectorDB = resources["vector_db"]
        response = db.index.fetch(ids=[doc_id], namespace=namespace)
        vectors = response.get("vectors", {})
        if doc_id not in vectors:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        return {"id": doc_id, "metadata": vectors[doc_id].get("metadata", {}), "namespace": namespace}
    except HTTPException: raise
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, namespace: str = ""):
    """Delete a single document by ID."""
    try:
        db: VectorDB = resources["vector_db"]
        db.index.delete(ids=[doc_id], namespace=namespace)
        return {"message": f"Document {doc_id} deleted."}
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest-file")
async def ingest_file(request: schemas.IngestFileRequest):
    """Load and index a JSON file from disk."""
    try:
        file_path = Config.BASE_DIR / request.file_path
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = [schemas.UpsertItem(id=str(e.get("id", e.get("name", "missing_id"))), text=e.get("description", e.get("text", "")), metadata=e) for e in data]
        return await upsert_data(schemas.UpsertRequest(items=items, namespace=request.namespace))
    except Exception as e:
        logger.error(f"File ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate")
async def evaluate_namespace(request: schemas.EvaluationRequest):
    """Run local evaluation script logic through the API."""
    try:
        from src.evaluation.evaluator import Evaluator
        from src.processing.data_processor import DataProcessor
        processor = DataProcessor()
        evaluator = Evaluator()
        queries_path = Config.BASE_DIR / request.queries_file
        if not queries_path.exists():
            raise HTTPException(status_code=404, detail="Queries file not found")
        queries = processor.load_json(str(queries_path))
        namespace = request.namespace or queries_path.stem.replace("test_queries_", "")
        df = evaluator.run(queries, namespace=namespace)
        metrics = {"hit@1": df["hit@1"].mean(), "hit@5": df["hit@5"].mean(), "mrr": df["mrr"].mean(), "total_queries": len(df)}
        return {"namespace": namespace, "metrics": metrics}
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results")
async def list_results():
    """List all CSV result files."""
    try:
        results_dir = Path(Config.RESULTS_DIR)
        files = glob.glob(str(results_dir / "*.csv"))
        return {"files": [os.path.basename(f) for f in files]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs")
async def get_logs(lines: int = 50):
    """Retrieve recent application logs."""
    try:
        log_file = Config.BASE_DIR / "app.log"
        if not log_file.exists(): return {"message": "Log file not found", "logs": []}
        with open(log_file, "r") as f:
            return {"logs": f.readlines()[-lines:]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config")
async def update_config(request: schemas.ConfigUpdateRequest):
    """Update runtime settings."""
    if request.alpha is not None: Config.ALPHA = request.alpha
    if request.top_k is not None: Config.TOP_K = request.top_k
    return {"message": "Config updated", "ALPHA": Config.ALPHA, "TOP_K": Config.TOP_K}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
