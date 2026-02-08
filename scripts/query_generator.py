"""
Query generation from synthetic tire factory entities.

Generates diverse test queries (exact, fuzzy, location, type, semantic)
for RAG retrieval benchmarking.
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config

INPUT_DIR = Config.SYNTHETIC_DATA_DIR
OUTPUT_DIR = Config.QUERIES_DATA_DIR

QUERY_PATTERNS = {
    "exact": [
        "show me {name}",
        "check {name}",
        "{name} status",
        "get info on {name}",
        "display {name}"
    ],
    "fuzzy": [
        "check {name_fuzzy}",
        "show {name_fuzzy}",
        "{name_fuzzy} info"
    ],
    "location": [
        "machines in {location}",
        "show {location} equipment",
        "what's in {location}",
        "{location} status",
        "list {location}"
    ],
    "type": [
        "show all {type}",
        "list {type}",
        "find {type}",
        "which {type}"
    ],
    "semantic": [
        "which has {keyword}",
        "find {keyword}",
        "machines with {keyword}"
    ]
}

TYPOS = {
    "MAXX": ["MAX", "MAAX"],
    "Press": ["Pres", "Prss"],
    "Curing": ["Cring", "Curring"],
    "Machine": ["Machne"]
}


def apply_typo(text: str) -> str:
    """
    Apply a random typo to text for fuzzy matching tests.
    
    Args:
        text: Original text
        
    Returns:
        Text with typo applied or spaces removed
    """
    for correct, typos in TYPOS.items():
        if correct in text:
            return text.replace(correct, random.choice(typos))
    return text.replace(" ", "")


def extract_location(path: str) -> str:
    """
    Extract location from entity path.
    
    Args:
        path: Full entity path
        
    Returns:
        Location string (second path component)
    """
    parts = path.split("/")
    if len(parts) > 2:
        return parts[1]
    return "Production Site A"


def extract_keywords(description: str) -> List[str]:
    """
    Extract relevant keywords from description for semantic queries.
    
    Args:
        description: Entity description
        
    Returns:
        List of up to 3 keywords found in description
    """
    if not description:
        return []
    
    target_keywords = [
        "pressure", "temperature", "control", "system", "sensor",
        "assembly", "vulcanization", "mixing", "testing"
    ]
    
    keywords = [
        word for word in target_keywords
        if word in description.lower()
    ]
    
    return keywords[:3]


def generate_queries_for_object(obj: Dict, count: int = 5) -> List[Dict]:
    """
    Generate diverse queries for a single entity object.
    
    Args:
        obj: Entity dictionary with name, type, path, description, guid
        count: Number of queries to generate
        
    Returns:
        List of query dictionaries
    """
    queries = []
    name = obj["name"]
    obj_type = obj["type_name"]
    location = extract_location(obj["path"])
    keywords = extract_keywords(obj.get("description", ""))
    
    # Define strategy distribution
    strategies = ["exact"] * 2 + ["fuzzy"] * 1 + ["location"] * 1 + ["type"] * 1
    if keywords:
        strategies.append("semantic")
    
    random.shuffle(strategies)
    
    for i, strategy in enumerate(strategies[:count]):
        query = _create_query_for_strategy(
            strategy=strategy,
            obj=obj,
            name=name,
            obj_type=obj_type,
            location=location,
            keywords=keywords,
            query_index=i
        )
        
        if query:
            queries.append(query)
    
    return queries


def _create_query_for_strategy(
    strategy: str,
    obj: Dict,
    name: str,
    obj_type: str,
    location: str,
    keywords: List[str],
    query_index: int
) -> Dict:
    """
    Create a query based on the specified strategy.
    
    Args:
        strategy: Query type strategy
        obj: Original entity object
        name: Entity name
        obj_type: Entity type
        location: Entity location
        keywords: Semantic keywords
        query_index: Index for query ID generation
        
    Returns:
        Query dictionary or None if strategy cannot be applied
    """
    query_id = f"Q_{obj['guid']}_{query_index + 1:02d}"
    
    if strategy == "exact":
        pattern = random.choice(QUERY_PATTERNS["exact"])
        query_text = pattern.format(name=name)
        return {
            "query_id": query_id,
            "query": query_text,
            "query_type": "exact_match",
            "difficulty": "easy",
            "expected_object_id": obj["guid"],
            "expected_object_name": name,
            "expected_object_type": obj_type
        }
    
    elif strategy == "fuzzy":
        name_fuzzy = apply_typo(name)
        pattern = random.choice(QUERY_PATTERNS["fuzzy"])
        query_text = pattern.format(name_fuzzy=name_fuzzy)
        return {
            "query_id": query_id,
            "query": query_text,
            "query_type": "fuzzy_match",
            "difficulty": "medium",
            "expected_object_id": obj["guid"],
            "expected_object_name": name,
            "expected_object_type": obj_type,
            "has_typo": True
        }
    
    elif strategy == "location":
        pattern = random.choice(QUERY_PATTERNS["location"])
        query_text = pattern.format(location=location)
        return {
            "query_id": query_id,
            "query": query_text,
            "query_type": "location_search",
            "difficulty": "hard",
            "expected_object_id": obj["guid"],
            "expected_object_name": name,
            "expected_object_type": obj_type,
            "note": "Multiple results expected"
        }
    
    elif strategy == "type":
        pattern = random.choice(QUERY_PATTERNS["type"])
        query_text = pattern.format(type=obj_type)
        return {
            "query_id": query_id,
            "query": query_text,
            "query_type": "type_search",
            "difficulty": "hard",
            "expected_object_id": obj["guid"],
            "expected_object_name": name,
            "expected_object_type": obj_type,
            "note": "Multiple results expected"
        }
    
    elif strategy == "semantic" and keywords:
        keyword = random.choice(keywords)
        pattern = random.choice(QUERY_PATTERNS["semantic"])
        query_text = pattern.format(keyword=keyword)
        return {
            "query_id": query_id,
            "query": query_text,
            "query_type": "semantic_search",
            "difficulty": "very_hard",
            "expected_object_id": obj["guid"],
            "expected_object_name": name,
            "expected_object_type": obj_type
        }
    
    return None


def generate_query_dataset(objects_file: str, queries_per_object: int = 5):
    """
    Generate complete query dataset from entity file.
    
    Args:
        objects_file: Filename of entity JSON file
        queries_per_object: Number of queries to generate per entity
    """
    filepath = INPUT_DIR / objects_file
    
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    
    with open(filepath, "r", encoding="utf-8") as f:
        objects = json.load(f)
    
    print(f"Loaded {len(objects)} objects from {objects_file}")
    
    all_queries = []
    for obj in objects:
        queries = generate_queries_for_object(obj, queries_per_object)
        all_queries.extend(queries)
    
    # Calculate distribution
    query_types = {}
    for query in all_queries:
        qtype = query["query_type"]
        query_types[qtype] = query_types.get(qtype, 0) + 1
    
    # Save to file
    output_file = objects_file.replace("synthetic_entities", "test_queries")
    output_path = OUTPUT_DIR / output_file
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_queries, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved {len(all_queries)} queries to {output_file}")
    print(f"Distribution: {query_types}\n")


def main():
    """Main entry point for query generation."""
    print("Query Generator\n")
    
    files = list(INPUT_DIR.glob("synthetic_entities_*.json"))
    
    if not files:
        print(f"No entity files found in {INPUT_DIR}")
        return
    
    print("Available files:")
    for i, file in enumerate(files, 1):
        print(f"  {i}. {file.name}")
    
    choice = input(f"\nChoose file (1-{len(files)}) or 'all': ").strip()
    
    queries_per_obj = int(input("Queries per object [5]: ").strip() or "5")
    
    if choice.lower() == "all":
        for file in files:
            generate_query_dataset(file.name, queries_per_obj)
    else:
        try:
            idx = int(choice) - 1
            generate_query_dataset(files[idx].name, queries_per_obj)
        except (ValueError, IndexError):
            print("Invalid choice")


if __name__ == "__main__":
    main()