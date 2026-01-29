import os
import json
import random
from typing import List, Dict
from pathlib import Path
from src.utils.config import Config

INPUT_DIR = Path(Config.SYNTHETIC_DATA_DIR)
OUTPUT_DIR = Path(Config.QUERIES_DATA_DIR)

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
    for correct, typos in TYPOS.items():
        if correct in text:
            return text.replace(correct, random.choice(typos))
    return text.replace(' ', '')

def extract_location(path: str) -> str:
    parts = path.split('/')
    if len(parts) > 2:
        return parts[1]
    return "Production Site A"

def extract_keywords(description: str) -> List[str]:
    if not description:
        return []
    keywords = []
    for word in ['pressure', 'temperature', 'control', 'system', 'sensor', 'assembly', 'vulcanization', 'mixing', 'testing']:
        if word in description.lower():
            keywords.append(word)
    return keywords[:3]

def generate_queries_for_object(obj: Dict, count: int = 5) -> List[Dict]:
    queries = []
    name = obj['name']
    obj_type = obj['type_name']
    location = extract_location(obj['path'])
    keywords = extract_keywords(obj.get('description', ''))
    
    strategies = ['exact'] * 2 + ['fuzzy'] * 1 + ['location'] * 1 + ['type'] * 1
    if keywords:
        strategies.append('semantic')
    
    random.shuffle(strategies)
    
    for i, strategy in enumerate(strategies[:count]):
        
        if strategy == 'exact':
            pattern = random.choice(QUERY_PATTERNS['exact'])
            query_text = pattern.format(name=name)
            query = {
                "query_id": f"Q_{obj['guid']}_{i+1:02d}",
                "query": query_text,
                "query_type": "exact_match",
                "difficulty": "easy",
                "expected_object_id": obj['guid'],
                "expected_object_name": name,
                "expected_object_type": obj_type
            }
        
        elif strategy == 'fuzzy':
            name_fuzzy = apply_typo(name)
            pattern = random.choice(QUERY_PATTERNS['fuzzy'])
            query_text = pattern.format(name_fuzzy=name_fuzzy)
            query = {
                "query_id": f"Q_{obj['guid']}_{i+1:02d}",
                "query": query_text,
                "query_type": "fuzzy_match",
                "difficulty": "medium",
                "expected_object_id": obj['guid'],
                "expected_object_name": name,
                "expected_object_type": obj_type,
                "has_typo": True
            }
        
        elif strategy == 'location':
            pattern = random.choice(QUERY_PATTERNS['location'])
            query_text = pattern.format(location=location)
            query = {
                "query_id": f"Q_{obj['guid']}_{i+1:02d}",
                "query": query_text,
                "query_type": "location_search",
                "difficulty": "hard",
                "expected_object_id": obj['guid'],
                "expected_object_name": name,
                "expected_object_type": obj_type,
                "note": "Multiple results expected"
            }
        
        elif strategy == 'type':
            pattern = random.choice(QUERY_PATTERNS['type'])
            query_text = pattern.format(type=obj_type)
            query = {
                "query_id": f"Q_{obj['guid']}_{i+1:02d}",
                "query": query_text,
                "query_type": "type_search",
                "difficulty": "hard",
                "expected_object_id": obj['guid'],
                "expected_object_name": name,
                "expected_object_type": obj_type,
                "note": "Multiple results expected"
            }
        
        elif strategy == 'semantic' and keywords:
            keyword = random.choice(keywords)
            pattern = random.choice(QUERY_PATTERNS['semantic'])
            query_text = pattern.format(keyword=keyword)
            query = {
                "query_id": f"Q_{obj['guid']}_{i+1:02d}",
                "query": query_text,
                "query_type": "semantic_search",
                "difficulty": "very_hard",
                "expected_object_id": obj['guid'],
                "expected_object_name": name,
                "expected_object_type": obj_type
            }
        
        else:
            continue
        
        queries.append(query)
    
    return queries

def generate_query_dataset(objects_file: str, queries_per_object: int = 5):
    filepath = INPUT_DIR / objects_file
    
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        objects = json.load(f)
    
    print(f"Loaded {len(objects)} objects from {objects_file}")
    
    all_queries = []
    for obj in objects:
        queries = generate_queries_for_object(obj, queries_per_object)
        all_queries.extend(queries)
    
    query_types = {}
    for q in all_queries:
        qtype = q['query_type']
        query_types[qtype] = query_types.get(qtype, 0) + 1
    
    output_file = objects_file.replace('synthetic_entities', 'test_queries')
    output_path = OUTPUT_DIR / output_file
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_queries, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved {len(all_queries)} queries to {output_file}")
    print(f"Distribution: {query_types}\n")

def main():
    print("Query Generator\n")
    
    files = list(INPUT_DIR.glob("synthetic_entities_*.json"))
    
    if not files:
        print(f"No entity files found in {INPUT_DIR}")
        return
    
    print("Available files:")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f.name}")
    
    choice = input(f"\nChoose file (1-{len(files)}) or 'all': ").strip()
    
    queries_per_obj = int(input("Queries per object [5]: ").strip() or "5")
    
    if choice.lower() == 'all':
        for f in files:
            generate_query_dataset(f.name, queries_per_obj)
    else:
        try:
            idx = int(choice) - 1
            generate_query_dataset(files[idx].name, queries_per_obj)
        except:
            print("Invalid choice")

if __name__ == "__main__":
    main()