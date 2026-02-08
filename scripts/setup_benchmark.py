"""
Benchmark verification script for golden dataset validation.

Validates that the golden benchmark dataset has the required structure
and columns for RAG evaluation.
"""

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Setup
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")


class BenchmarkValidationError(Exception):
    """Raised when benchmark validation fails."""
    pass


def verify_benchmark():
    """
    Verify the golden benchmark dataset structure and content.
    
    Raises:
        BenchmarkValidationError: If validation fails
    """
    print(f"\n{'=' * 50}")
    print("BENCHMARK VERIFICATION")
    print(f"{'=' * 50}")

    # Define file path
    input_file = project_root / "datasets" / "golden_descriptions_10.json"
    
    if not input_file.exists():
        raise BenchmarkValidationError(f"File not found: {input_file}")

    # Load golden dataset
    print(f"Loading {input_file.name}...")
    
    try:
        df_golden = pd.read_json(input_file)
        print(f"Loaded {len(df_golden)} rows.")
        
        # Verify required columns
        required_cols = ["name", "description", "audit_label"]
        missing_cols = [col for col in required_cols if col not in df_golden.columns]
        
        if missing_cols:
            raise BenchmarkValidationError(
                f"Missing required columns in JSON: {missing_cols}"
            )
        
        # Check for empty dataset
        if len(df_golden) == 0:
            raise BenchmarkValidationError("Dataset is empty")
        
        print("Dataset structure validated successfully.")
        print(f"Columns: {list(df_golden.columns)}")
        print(f"Rows: {len(df_golden)}")
        print("\nGolden Benchmark dataset is valid and ready.")
        
    except pd.errors.EmptyDataError as e:
        raise BenchmarkValidationError(f"Empty or invalid JSON file: {e}") from e
    except ValueError as e:
        raise BenchmarkValidationError(f"JSON parsing error: {e}") from e


def main():
    """Main entry point for benchmark verification."""
    try:
        verify_benchmark()
        return 0
    except BenchmarkValidationError as e:
        print(f"\nValidation failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
