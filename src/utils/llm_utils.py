"""Shared LLM utility functions for JSON extraction and response parsing."""

import json
import re
from typing import Any, Optional


class JSONExtractionError(Exception):
    """Raised when JSON extraction from LLM response fails."""
    pass


def extract_json_from_response(text: str) -> Any:
    """
    Extract and parse JSON from LLM response text.
    
    Handles multiple formats:
    - Markdown code blocks with ```json ... ```
    - Generic code blocks with ``` ... ```
    - Raw JSON strings
    
    Args:
        text: Raw text from LLM response
        
    Returns:
        Parsed JSON object (dict, list, etc.)
        
    Raises:
        JSONExtractionError: If JSON cannot be extracted or parsed
    """
    if not text or not isinstance(text, str):
        raise JSONExtractionError("Input text is empty or not a string")
    
    text = text.strip()
    
    # Try to extract from markdown code blocks
    json_match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL | re.IGNORECASE)
    if json_match:
        json_str = json_match.group(1).strip()
        return _parse_json(json_str)
    
    # Try generic code blocks
    code_match = re.search(r'```\s*\n(.*?)\n```', text, re.DOTALL)
    if code_match:
        json_str = code_match.group(1).strip()
        return _parse_json(json_str)
    
    # Try parsing raw text as JSON
    return _parse_json(text)


def _parse_json(json_str: str) -> Any:
    """
    Parse JSON string with error handling.
    
    Args:
        json_str: String containing JSON data
        
    Returns:
        Parsed JSON object
        
    Raises:
        JSONExtractionError: If parsing fails
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise JSONExtractionError(
            f"Failed to parse JSON: {e.msg} at line {e.lineno}, column {e.colno}"
        ) from e


def validate_json_structure(
    data: Any,
    expected_type: type,
    required_keys: Optional[list] = None
) -> bool:
    """
    Validate the structure of parsed JSON data.
    
    Args:
        data: Parsed JSON data
        expected_type: Expected type (list, dict, etc.)
        required_keys: List of required keys (for dict validation)
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(data, expected_type):
        raise ValueError(
            f"Expected type {expected_type.__name__}, got {type(data).__name__}"
        )
    
    if expected_type == dict and required_keys:
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")
    
    if expected_type == list and required_keys:
        for item in data:
            if isinstance(item, dict):
                missing_keys = [key for key in required_keys if key not in item]
                if missing_keys:
                    raise ValueError(
                        f"Item missing required keys: {missing_keys}"
                    )
    
    return True
