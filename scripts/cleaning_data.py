"""
Data cleaning script for raw entity data.

Loads raw entities, processes them, and saves cleaned output.
"""

from src.processing.data_processor import DataProcessor
from src.utils.config import Config


def main():
    """Main execution for data cleaning."""
    processor = DataProcessor()
    
    # Define paths using pathlib
    input_path = Config.RAW_DATA_DIR / "entities.json"
    output_path = Config.PROCESSED_DATA_DIR / "cleaned_entities.json"
    
    print(f"Loading raw data from: {input_path}")
    raw_data = processor.load_json(str(input_path))
    
    # Process entities
    # Note: If entities is a dict with 'entities' key, extract it
    entities_list = (
        raw_data.get("entities", [])
        if isinstance(raw_data, dict)
        else raw_data
    )
    
    cleaned_data = processor.process_raw_entities(entities_list)
    
    print(f"Cleaned {len(cleaned_data)} entities.")
    
    # Save results
    processor.save_json(cleaned_data, str(output_path))


if __name__ == "__main__":
    main()