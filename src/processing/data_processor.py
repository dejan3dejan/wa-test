"""Data processing utilities for loading, saving, and cleaning entity data."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataProcessingError(Exception):
    """Raised when data processing operations fail."""
    pass


class DataProcessor:
    """Handles loading, saving, and processing of entity data."""
    
    @staticmethod
    def load_json(file_path: Union[str, Path]) -> List[Dict]:
        """
        Load JSON data from file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded data as list of dictionaries
            
        Raises:
            DataProcessingError: If loading fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Ensure data is a list
            if not isinstance(data, list):
                logger.warning(
                    f"Expected list but got {type(data).__name__}. "
                    f"Wrapping in list."
                )
                data = [data] if data else []
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            return []
        except IOError as e:
            logger.error(f"IO error reading {file_path}: {e}")
            return []

    @staticmethod
    def save_json(data: Any, file_path: Union[str, Path]):
        """
        Save data to JSON file.
        
        Args:
            data: Data to save (typically list or dict)
            file_path: Output file path
            
        Raises:
            DataProcessingError: If saving fails
        """
        file_path = Path(file_path)
        
        try:
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Data successfully saved to: {file_path}")
            
        except (IOError, OSError) as e:
            raise DataProcessingError(
                f"Error saving to {file_path}: {e}"
            ) from e
        except TypeError as e:
            raise DataProcessingError(
                f"Data is not JSON serializable: {e}"
            ) from e

    @staticmethod
    def clean_description(entity: Dict) -> str:
        """
        Extract and clean description from entity.
        
        Tries entity description first, falls back to type description.
        
        Args:
            entity: Entity dictionary
            
        Returns:
            Cleaned description string (empty if none found)
        """
        # Try entity description
        desc = entity.get("description")
        if desc and isinstance(desc, str) and desc.strip():
            return desc.strip()
        
        # Fall back to type description
        entity_type = entity.get("type")
        if entity_type and isinstance(entity_type, dict):
            type_desc = entity_type.get("description")
            if type_desc and isinstance(type_desc, str) and type_desc.strip():
                return type_desc.strip()
        
        return ""

    @classmethod
    def process_raw_entities(cls, raw_data: List[Dict]) -> List[Dict]:
        """
        Process and clean raw entity data.
        
        Args:
            raw_data: List of raw entity dictionaries
            
        Returns:
            List of cleaned entity dictionaries
        """
        cleaned_entities = []
        skipped = 0
        
        for entity in raw_data:
            guid = entity.get("guid")
            name = entity.get("name", "").strip()
            
            # Skip entities without GUID or name
            if not guid or not name:
                skipped += 1
                continue
            
            description = cls.clean_description(entity)
            
            # Skip folder entities without descriptions
            if name.lower() in ["devices"] and not description:
                skipped += 1
                continue
            
            # Extract type information
            entity_type = entity.get("type")
            type_name = ""
            if entity_type and isinstance(entity_type, dict):
                type_name = entity_type.get("name", "")
            
            cleaned_entities.append({
                "guid": guid,
                "name": name,
                "description": description,
                "path": entity.get("path", ""),
                "type_name": type_name,
                "is_synthetic": entity.get("is_synthetic", False)
            })
        
        if skipped > 0:
            logger.info(f"Skipped {skipped} entities during cleaning")
        
        return cleaned_entities