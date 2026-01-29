import os
from src.utils.config import Config
from src.processing.data_processor import DataProcessor

def main():
    """Main execution for data cleaning."""
    processor = DataProcessor()
    
    # Define paths
    input_path = os.path.join(Config.RAW_DATA_DIR, "entities.json")
    output_path = os.path.join(Config.PROCESSED_DATA_DIR, "cleaned_entities.json")
    
    print(f"Loading raw data from: {input_path}")
    raw_data = processor.load_json(input_path)
    
    # Process entities (using the logic we moved to DataProcessor)
    # Note: If entities is a dict with 'entities' key, extract it
    entities_list = raw_data.get('entities', []) if isinstance(raw_data, dict) else raw_data
    
    cleaned_data = processor.process_raw_entities(entities_list)
    
    print(f"Cleaned {len(cleaned_data)} entities.")
    
    # Save results
    processor.save_json(cleaned_data, output_path)

if __name__ == "__main__":
    main()