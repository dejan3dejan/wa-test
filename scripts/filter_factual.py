import json
import sys
from pathlib import Path

def filter_factual(input_file: str, output_file: str):
    """
    Filter only factual items from audited dataset.
    
    Args:
        input_file: Path to audited JSON
        output_file: Path to save filtered factual items
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    factual_items = [item for item in data if item.get('audit_label') == 'factual']
    hallucinated_items = [item for item in data if item.get('audit_label') == 'hallucinated']
    
    print(f"Total items: {len(data)}")
    print(f"Factual: {len(factual_items)} ({len(factual_items)/len(data)*100:.1f}%)")
    print(f"Hallucinated: {len(hallucinated_items)} ({len(hallucinated_items)/len(data)*100:.1f}%)")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(factual_items, f, indent=2, ensure_ascii=False)
    
    print(f"\nFactual items saved to: {output_file}")
    return len(factual_items)

if __name__ == "__main__":
    input_path = "data/synthetic/synthetic_entities_1000_mixed_audited.json"
    output_path = "data/synthetic/synthetic_entities_1000_factual.json"
    
    filter_factual(input_path, output_path)
