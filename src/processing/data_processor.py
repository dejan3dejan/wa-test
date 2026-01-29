import json
import os
from typing import List, Dict

class DataProcessor:
    @staticmethod
    def load_json(file_path: str) -> List[Dict]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []

    @staticmethod
    def save_json(data: List[Dict], file_path: str):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"Data successfully saved in: {file_path}")
        except Exception as e:
            print(f"Error saving to {file_path}: {e}")

    @staticmethod
    def clean_description(entity: Dict) -> str:
        desc = entity.get('description')
        if desc and desc.strip():
            return desc.strip()
        
        entity_type = entity.get('type')
        if entity_type and isinstance(entity_type, dict):
            type_desc = entity_type.get('description')
            if type_desc and type_desc.strip():
                return type_desc.strip()
        
        return ""

    @classmethod
    def process_raw_entities(cls, raw_data: List[Dict]) -> List[Dict]:
        cleaned_entities = []
        for entity in raw_data:
            guid = entity.get('guid')
            name = entity.get('name', '').strip()
            
            if not guid or not name:
                continue
            
            description = cls.clean_description(entity)
            
            # Skip folders without descriptions
            if name.lower() in ['devices'] and not description:
                continue
            
            cleaned_entities.append({
                "guid": guid,
                "name": name,
                "description": description,
                "path": entity.get('path', ''),
                "type_name": entity.get('type', {}).get('name', '') if entity.get('type') else '',
                "is_synthetic": entity.get('is_synthetic', False)
            })
        return cleaned_entities