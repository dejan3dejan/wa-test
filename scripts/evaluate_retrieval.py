import os
import pandas as pd
from src.utils.config import Config
from src.processing.data_processor import DataProcessor
from src.evaluation.evaluator import Evaluator

def run_evaluation(queries_file):
    """Entry point for the evaluation process."""
    processor = DataProcessor()
    evaluator = Evaluator()

    # 1. Loading queries using DataProcessor
    queries = processor.load_json(queries_file)
    print(f"Evaluating {len(queries)} queries from {queries_file}\n")
    
    # 2. Running evaluation logic using the Evaluator class
    df = evaluator.run(queries)

    # 3. Printing summary metrics to console
    print("\n" + "="*30)
    print("RESULTS")
    print("="*30)
    print(f"Hit@1:  {df['hit@1'].mean():.2%}")
    print(f"Hit@5:  {df['hit@5'].mean():.2%}")
    print(f"MRR:    {df['mrr'].mean():.4f}")
    
    # 4. Saving the CSV report to results/ directory
    base_name = os.path.basename(queries_file).replace(".json", ".csv")
    output_path = os.path.join(Config.RESULTS_DIR, f"eval_results_{base_name}")
    
    df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved in: {output_path}")

if __name__ == "__main__":
    # Point to the specific query file you want to test
    QUERIES_PATH = os.path.join(Config.QUERIES_DATA_DIR, "test_queries_10_mixed.json")
    run_evaluation(QUERIES_PATH)