import os
import json
import random
import time
from google import genai
from typing import List, Dict
from dotenv import load_dotenv
from pathlib import Path
from src.utils.config import Config

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.5-flash"

OUTPUT_DIR = Path(Config.SYNTHETIC_DATA_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET_SIZES = [10, 100, 1000]

PROFILES = {
    "realistic": {"empty": 0.40, "minimal": 0.30, "medium": 0.20, "detailed": 0.10},
    "clean": {"empty": 0.05, "minimal": 0.20, "medium": 0.40, "detailed": 0.35},
    "mixed": {"empty": 0.20, "minimal": 0.25, "medium": 0.35, "detailed": 0.20}
}

def load_seeds(path: str = None) -> List[Dict]:
    if path is None:
        path = os.path.join(Config.PROCESSED_DATA_DIR, 'cleaned_entities.json')
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

seeds = load_seeds()

def generate_batch(size: int, profile: str = "realistic") -> List[Dict]:
    if not seeds:
        return []
    
    dist = PROFILES[profile]
    target_empty = int(size * dist["empty"])
    target_minimal = int(size * dist["minimal"])
    target_medium = int(size * dist["medium"])
    target_detailed = size - (target_empty + target_minimal + target_medium)
    
    examples = random.sample(seeds, min(3, len(seeds)))
    
    prompt = f"""
Generate {size} tire factory entities.

DISTRIBUTION:
- {target_empty} with description: ""
- {target_minimal} with 5-10 word descriptions
- {target_medium} with 2-3 sentence descriptions
- {target_detailed} with detailed descriptions

EXAMPLES:

Empty:
{{"name": "VMI EDGER 1", "description": "", "type_name": "VMI EDGER", "path": "Smart Tire Production/Devices/VMI EDGER 1"}}
{{"name": "TUM 1", "description": "", "type_name": "Tire Uniformity Machine (TUM)", "path": "Smart Tire Production/Production Site A/Tire Production Line 1/TUM 1"}}

Minimal:
{{"name": "VMI MAXX 1", "description": "VMI MAXX high performance smart tire machine.", "type_name": "VMI MAXX", "path": "Smart Tire Production/Devices/VMI MAXX 1"}}

Medium:
{{"name": "Curing Press 1", "description": "Parallel processing bank of vulcanization presses. Acts as the line's primary bottleneck. Converts 'Green Tires' to finished tires using heat and pressure.", "type_name": "Curing Press", "path": "Smart Tire Production/Production Site A/Tire Production Line 1/Curing Press 1"}}

Detailed:
{{"name": "Tire Machine 1", "description": "High-speed discrete automation asset that assembles 'Green Tires' from uncured rubber components. Process Type: Discrete, Cycle-based (~40s). Key Physics: Tension control and mechanical alignment determine tire uniformity.", "type_name": "Smart Tire Machine", "path": "Smart Tire Production/Production Site A/Tire Production Line 1/Tire Machine 1"}}

Types (only use these):
VMI MAXX, VMI EDGER, Curing Press, Smart Tire Machine, Tire Uniformity Machine (TUM), Mixer, Cutter, Production Site, Production Line, Mixing Department, Logistics, AGV, Boiler System

Paths format: Smart Tire Production/[Location]/[Sublocation]/[Name]
Locations: Production Site A, Production Site B, Tire Production Line 1/2/3, Mixing Department, Logistics, Devices

full_text: If description empty use just name, else "name - description"

Output ONLY JSON array:
[{{"name": "...", "description": "...", "type_name": "...", "path": "...", "full_text": "..."}}]
"""
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            text = response.text.strip()
            
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()
            
            entities = json.loads(text)
            
            if isinstance(entities, list) and len(entities) == size:
                empty = sum(1 for e in entities if not e.get('description', '').strip())
                print(f"   Generated: {empty}/{target_empty} empty")
                if abs(empty - target_empty) <= 2:
                    return entities
                    
        except Exception as e:
            print(f"Error attempt {attempt+1}: {e}")
            time.sleep(3 + attempt * 3)
    
    return []

def generate_dataset(total: int, profile: str, batch_size: int = 10) -> List[Dict]:
    entities = []
    
    while len(entities) < total:
        remaining = total - len(entities)
        current = min(batch_size, remaining)
        batch = generate_batch(current, profile)
        
        if not batch:
            time.sleep(5)
            continue
        
        entities.extend(batch)
        print(f"Progress: {len(entities)}/{total}")
        time.sleep(2)
    
    entities = entities[:total]
    for i, e in enumerate(entities):
        e["guid"] = f"SYNTH_{profile.upper()}_{i+1:04d}"
        e["is_synthetic"] = True
    
    return entities

def save_dataset(entities: List[Dict], size: int, profile: str):
    filename = OUTPUT_DIR / f"synthetic_entities_{size}_{profile}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    
    empty = sum(1 for e in entities if not e.get('description', '').strip())
    minimal = sum(1 for e in entities if e.get('description', '').strip() and len(e['description']) < 100)
    medium = sum(1 for e in entities if 100 <= len(e.get('description', '')) < 300)
    detailed = sum(1 for e in entities if len(e.get('description', '')) >= 300)
    
    print(f"\nSaved: {filename}")
    print(f"Empty: {empty} ({empty/size*100:.1f}%), Minimal: {minimal} ({minimal/size*100:.1f}%), Medium: {medium} ({medium/size*100:.1f}%), Detailed: {detailed} ({detailed/size*100:.1f}%)\n")

def main():
    print("Dataset Generator - Realistic & Clean profiles\n")
    
    profile = input("Profile (realistic/clean/mixed) [realistic]: ").strip().lower() or "realistic"
    
    if profile not in PROFILES:
        profile = "realistic"
    
    print(f"\nGenerating with profile: {profile}\n")
    
    for size in DATASET_SIZES:
        print(f"=== Dataset {size} ===")
        try:
            batch_size = 20 if size >= 100 else 10
            entities = generate_dataset(size, profile, batch_size)
            save_dataset(entities, size, profile)
        except KeyboardInterrupt:
            print("\nInterrupted")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()