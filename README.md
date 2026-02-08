# Industrial RAG Benchmarking System

Retrieval-Augmented Generation (RAG) evaluation framework for industrial data, featuring hybrid search with dense semantic embeddings and sparse keyword matching.

## Overview

This system implements a comprehensive RAG benchmarking pipeline designed to evaluate retrieval performance in industrial settings. It combines Google Gemini for semantic embeddings with Pinecone's vector database, enhanced by BM25 sparse embeddings for optimal hybrid search performance.

**Key Features:**
- Hybrid search architecture (Dense + Sparse BM25)
- Dependency injection for testability
- Custom exception handling and robust error management
- Comprehensive evaluation metrics (Hit@K, MRR)
- Synthetic dataset generation with multiple quality profiles
- Industrial domain focus (Smart Tire Production entities)

## Architecture

### System Components

```
src/
├── database/
│   └── vector_db.py          # Pinecone interface with hybrid search
├── evaluation/
│   └── evaluator.py          # Evaluation metrics (Hit@K, MRR)
├── processing/
│   ├── embedder.py           # Dense (Gemini) + Sparse (BM25) embeddings
│   └── data_processor.py     # Data loading, cleaning, and processing
└── utils/
    ├── config.py             # Centralized configuration
    ├── llm_utils.py          # LLM response parsing utilities
    └── logger.py             # Logging setup

scripts/
├── generate_test_queries.py  # Generate synthetic entities with Gemini
├── query_generator.py        # Generate test queries from entities
├── upsert_to_pinecone.py     # Upsert vectors to Pinecone with BM25 fitting
├── evaluate_retrieval.py     # Run evaluation and generate reports
├── audit_generator.py        # Audit synthetic data quality
└── setup_benchmark.py        # Validate benchmark datasets
```

### Architecture Diagrams

[View interactive diagrams on Eraser](https://app.eraser.io/workspace/TXfnAgvQnjOJGXvuS5Lq?origin=share)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_index_name_here
GEMINI_API_KEY=your_gemini_api_key_here
```

**Note:** Use `.env.example` as a template.

## Usage

### Generate Synthetic Entities

```bash
python scripts/generate_test_queries.py
```

Generates tire factory entities with varying description quality:
- **Realistic**: 40% empty, 30% minimal, 20% medium, 10% detailed
- **Clean**: 5% empty, 20% minimal, 40% medium, 35% detailed
- **Mixed**: 20% empty, 25% minimal, 35% medium, 20% detailed

### Generate Test Queries

```bash
python scripts/query_generator.py
```

Creates diverse query types:
- **Exact Match**: Direct name queries
- **Fuzzy Match**: Queries with typos
- **Location Search**: Queries by factory location
- **Type Search**: Queries by equipment type
- **Semantic Search**: Keyword-based queries

### Upsert to Pinecone

```bash
python scripts/upsert_to_pinecone.py
```

- Fits BM25 encoder on document corpus
- Generates dense embeddings (Gemini) and sparse embeddings (BM25)
- Upserts vectors to Pinecone with namespace partitioning
- Saves BM25 models for evaluation

### Run Evaluation

```bash
python scripts/evaluate_retrieval.py
```

- Loads pre-fitted BM25 models
- Performs hybrid search queries
- Computes evaluation metrics
- Saves detailed results to `results/`

## Evaluation Methodology

The system measures retrieval precision through:

- **Hit@1, Hit@3, Hit@5, Hit@10**: Percentage of queries where the correct result appears in top N
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of correct results
- **Breakdowns**: Performance by query type and difficulty

All reports are automatically generated in the `results/` directory as CSV files.

## Benchmark Results

Comprehensive evaluation across three dataset types (clean, mixed, realistic) and three sizes (10, 100, 1000 examples) revealed the following patterns:

### Performance by Dataset Size

| Size    | Hit@1 (Clean) | Hit@1 (Mixed) | Hit@1 (Realistic) | Average MRR |
|---------|---------------|---------------|-------------------|-------------|
| **10**  | 10.0%         | 10.0%         | 10.0%             | 0.293       |
| **100** | 27.0%         | **32.4%**     | 26.0%             | 0.434       |
| **1000**| 8.2%          | 8.2%          | **5.6%**          | 0.138       |

### Key Findings

**Optimal Scale: ~100 Examples**
- System achieves best performance at this scale
- Mixed dataset yields highest Hit@1 (32.4%)

**Scalability Challenge at 1000+ Examples**
- Performance drops 70-80% compared to 100 examples
- Median rank jumps from 3 to 999 (maximum possible)
- 67-77% of queries rank correct result at last position
- System often >80% confident in wrong results

**Performance by Query Type (1000 examples)**
- Exact/Fuzzy Match: 13-15% Hit@1 (easiest queries)
- Type Search: 1-2% Hit@1
- Location Search: 0.2-0.5% Hit@1
- Semantic Search: 0.5-0.8% Hit@1

### Statistical Significance
- Performance drop 100→1000 is highly statistically significant (p < 10⁻⁹⁰)
- Pattern is consistent across all dataset types
- Correlation correct_score vs rank: r ≈ -0.99

## Recommended Improvements

### 1. Cross-Encoder Re-ranking (Accuracy Boost)
- Use a cross-encoder model (e.g., BGE-Reranker) to re-score the top 50-100 candidates from initial retrieval.
- Significantly improves precision on dense datasets (e.g., 1000+ entities) by analyzing deep semantic alignment between query and candidate.

### 2. Metadata-Guided Retrieval (Hard Filtering)
- Implement LLM-based query parsing to extract hard filters (e.g., `location`, `equipment_type`).
- Apply these filters directly in Pinecone before semantic search to eliminate noise from unrelated departments or sites.

### 3. GraphRAG (Hierarchical Reasoning)
- Transition to a Knowledge Graph structure to preserve industrial hierarchies (Site → Line → Asset).
- Enables multi-hop reasoning (e.g., finding all components of a specific production line even if not mentioned in the machine's direct description). Note: Higher complexity/cost but superior for complex relationship queries.

### 4. Hybrid Search Tuning
- Further adjust the `ALPHA` parameter to balance dense (semantic) and sparse (keyword) signals based on query type (Exact vs. Fuzzy).

### 5. Metadata Augmentation
- Pre-process raw descriptions to include explicit contextual signals (e.g., `[TYPE: Mixer] [LOC: Site A]`).

## Development

### Project Structure

- `data/` - Local data storage (ignored by git)
  - `bm25_models/` - Fitted BM25 models per namespace
  - `synthetic/` - Generated synthetic entities
  - `queries/` - Generated test queries
  - `processed/` - Cleaned entity data
- `datasets/` - Golden datasets (ignored by git)
- `results/` - Evaluation outputs (ignored by git)
- `docs/` - Additional documentation

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest

# With coverage
pytest --cov=src tests/
```

### Code Style

The codebase follows PEP 8 guidelines with:
- Type hints for function signatures
- Comprehensive docstrings (Google/NumPy style)
- Custom exception classes for specific errors
- Dependency injection for testability

## License

MIT licence

## Acknowledgments

- Google Gemini for semantic embeddings
- Pinecone for vector database infrastructure
- BM25 for sparse keyword matching

---

**Note:** This is a research/benchmarking tool. Performance characteristics may vary based on your specific use case and data distribution.