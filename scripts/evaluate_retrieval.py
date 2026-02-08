"""
Evaluate retrieval performance on query datasets.

Runs RAG evaluation against Pinecone namespaces, computes metrics,
and saves detailed results to CSV files.
"""

import glob
import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.evaluator import Evaluator
from src.processing.data_processor import DataProcessor
from src.utils.config import Config


class EvaluationError(Exception):
    """Raised when evaluation fails."""
    pass


def run_evaluation(queries_file: str, namespace: str = None):
    """
    Run evaluation on a query file.
    
    Args:
        queries_file: Path to query JSON file
        namespace: Optional namespace override. If None, derives from filename
        
    Raises:
        EvaluationError: If evaluation fails
    """
    processor = DataProcessor()
    evaluator = Evaluator()

    # Determine namespace from filename if not provided
    if namespace is None:
        base_name = os.path.basename(queries_file)
        namespace = base_name.replace("test_queries_", "").replace(".json", "")

    # Load queries
    queries = processor.load_json(queries_file)
    
    if not queries:
        raise EvaluationError(f"No queries found in {queries_file}")
    
    print(f"\n{'=' * 70}")
    print(f"Evaluating: {os.path.basename(queries_file)}")
    print(f"Namespace: '{namespace}'")
    print(f"Queries: {len(queries)}")
    print(f"{'=' * 70}")
    
    # Run evaluation
    df = evaluator.run(queries, namespace=namespace)

    # Print summary metrics
    print("\n" + "=" * 70)
    print(f"RESULTS - {namespace}")
    print("=" * 70)
    print(f"Hit@1:  {df['hit@1'].mean():.2%}")
    print(f"Hit@3:  {df['hit@3'].mean():.2%}")
    print(f"Hit@5:  {df['hit@5'].mean():.2%}")
    print(f"Hit@10: {df['hit@10'].mean():.2%}")
    print(f"MRR:    {df['mrr'].mean():.4f}")
    
    # Breakdown by query type
    print("\nBreakdown by Query Type:")
    type_metrics = df.groupby("type").agg({
        "hit@1": "mean",
        "hit@5": "mean",
        "mrr": "mean"
    }).round(4)
    print(type_metrics)
    
    # Breakdown by difficulty
    print("\nBreakdown by Difficulty:")
    diff_metrics = df.groupby("difficulty").agg({
        "hit@1": "mean",
        "hit@5": "mean",
        "mrr": "mean"
    }).round(4)
    print(diff_metrics)
    
    # Save results to CSV
    results_dir = project_root / Config.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = os.path.basename(queries_file).replace(".json", ".csv")
    output_path = results_dir / f"eval_results_{base_name}"
    
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved: {output_path}")
    print("=" * 70 + "\n")


def main():
    """Main entry point for evaluating all query files."""
    queries_dir = project_root / Config.QUERIES_DATA_DIR
    pattern = str(queries_dir / "test_queries_*.json")
    query_files = glob.glob(pattern)
    
    if not query_files:
        print(f"No query files found in {queries_dir}")
        return
    
    print(f"\nFound {len(query_files)} query files for evaluation.\n")
    
    successful = 0
    failed = 0
    
    for file_path in sorted(query_files):
        try:
            run_evaluation(file_path)
            successful += 1
        except EvaluationError as e:
            print(f"Evaluation error for {file_path}: {e}\n")
            failed += 1
        except Exception as e:
            print(f"Unexpected error evaluating {file_path}: {e}\n")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"All evaluations finished!")
    print(f"Successful: {successful}/{len(query_files)}")
    print(f"Failed: {failed}/{len(query_files)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()