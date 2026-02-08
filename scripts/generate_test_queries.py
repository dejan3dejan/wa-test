"""
Synthetic entity generation for RAG benchmarking.

Generates tire factory entities with varying description quality using Gemini LLM.
Supports realistic, clean, and mixed profiles for diverse testing scenarios.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from google import genai

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.llm_utils import extract_json_from_response, JSONExtractionError

load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=Config.GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"

OUTPUT_DIR = Config.SYNTHETIC_DATA_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_SIZES = [10, 100, 1000]

PROFILES = {
    "realistic": {"empty": 0.40, "minimal": 0.30, "medium": 0.20, "detailed": 0.10},
    "clean": {"empty": 0.05, "minimal": 0.20, "medium": 0.40, "detailed": 0.35},
    "mixed": {"empty": 0.20, "minimal": 0.25, "medium": 0.35, "detailed": 0.20}
}


class EntityGenerationError(Exception):
    """Raised when entity generation fails."""
    pass


def load_seeds(path: str = None) -> List[Dict]:
    """
    Load seed entities from file for few-shot prompting.
    
    Args:
        path: Path to seed entities file. Defaults to cleaned_entities.json
        
    Returns:
        List of seed entity dictionaries
    """
    if path is None:
        path = Config.PROCESSED_DATA_DIR / "cleaned_entities.json"
    
    if not path.exists():
        return []
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


seeds = load_seeds()


def generate_batch(size: int, profile: str = "realistic") -> List[Dict]:
    """
    Generate a batch of synthetic entities using Gemini.
    
    Args:
        size: Number of entities to generate
        profile: Distribution profile (realistic, clean, mixed)
        
    Returns:
        List of generated entity dictionaries
        
    Raises:
        EntityGenerationError: If generation fails after retries
    """
    if not seeds:
        raise EntityGenerationError("No seed entities available for prompting")
    
    dist = PROFILES[profile]
    target_empty = int(size * dist["empty"])
    target_minimal = int(size * dist["minimal"])
    target_medium = int(size * dist["medium"])
    target_detailed = size - (target_empty + target_minimal + target_medium)
    
    prompt = _build_generation_prompt(
        size, target_empty, target_minimal, target_medium, target_detailed
    )
    
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME, contents=prompt
            )
            text = response.text.strip()
            
            entities = _parse_and_validate_batch(
                text, size, target_empty
            )
            
            if entities:
                return entities
                    
        except (JSONExtractionError, ValueError) as e:
            print(f"Error attempt {attempt + 1}: {e}")
            time.sleep(3 + attempt * 3)
    
    raise EntityGenerationError(
        f"Failed to generate valid batch after 3 attempts"
    )


def _build_generation_prompt(
    size: int,
    target_empty: int,
    target_minimal: int,
    target_medium: int,
    target_detailed: int
) -> str:
    """
    Build the LLM prompt for entity generation.
    
    Args:
        size: Total entities to generate
        target_empty: Number with empty descriptions
        target_minimal: Number with minimal descriptions
        target_medium: Number with medium descriptions
        target_detailed: Number with detailed descriptions
        
    Returns:
        Formatted prompt string
    """
    return f"""
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


def _parse_and_validate_batch(
    text: str,
    expected_size: int,
    target_empty: int
) -> List[Dict]:
    """
    Parse and validate generated entity batch.
    
    Args:
        text: Raw LLM response text
        expected_size: Expected number of entities
        target_empty: Target number of empty descriptions
        
    Returns:
        List of entities if valid, None otherwise
    """
    entities = extract_json_from_response(text)
    
    if not isinstance(entities, list) or len(entities) != expected_size:
        return None
    
    empty_count = sum(
        1 for entity in entities
        if not entity.get("description", "").strip()
    )
    
    print(f"   Generated: {empty_count}/{target_empty} empty")
    
    # Allow +/- 2 entities tolerance for empty count
    if abs(empty_count - target_empty) <= 2:
        return entities
    
    return None


def generate_dataset(
    total: int,
    profile: str,
    batch_size: int = 10
) -> List[Dict]:
    """
    Generate complete dataset of synthetic entities.
    
    Args:
        total: Total number of entities to generate
        profile: Distribution profile
        batch_size: Batch size for generation
        
    Returns:
        List of generated entities with GUIDs assigned
    """
    entities = []
    
    while len(entities) < total:
        remaining = total - len(entities)
        current_batch_size = min(batch_size, remaining)
        
        try:
            batch = generate_batch(current_batch_size, profile)
            entities.extend(batch)
            print(f"Progress: {len(entities)}/{total}")
            time.sleep(2)
        except EntityGenerationError as e:
            print(f"Batch generation failed: {e}")
            time.sleep(5)
    
    # Trim to exact size and assign GUIDs
    entities = entities[:total]
    for i, entity in enumerate(entities):
        entity["guid"] = f"SYNTH_{profile.upper()}_{i + 1:04d}"
        entity["is_synthetic"] = True
    
    return entities


def save_dataset(entities: List[Dict], size: int, profile: str):
    """
    Save generated dataset to file with statistics.
    
    Args:
        entities: List of entities to save
        size: Dataset size for filename
        profile: Profile name for filename
    """
    filename = OUTPUT_DIR / f"synthetic_entities_{size}_{profile}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    
    # Calculate distribution statistics
    empty = sum(
        1 for e in entities if not e.get("description", "").strip()
    )
    minimal = sum(
        1 for e in entities
        if e.get("description", "").strip() and len(e["description"]) < 100
    )
    medium = sum(
        1 for e in entities
        if 100 <= len(e.get("description", "")) < 300
    )
    detailed = sum(
        1 for e in entities if len(e.get("description", "")) >= 300
    )
    
    print(f"\nSaved: {filename}")
    print(
        f"Empty: {empty} ({empty / size * 100:.1f}%), "
        f"Minimal: {minimal} ({minimal / size * 100:.1f}%), "
        f"Medium: {medium} ({medium / size * 100:.1f}%), "
        f"Detailed: {detailed} ({detailed / size * 100:.1f}%)\n"
    )


def main():
    """Main entry point for dataset generation."""
    print("Dataset Generator - Realistic & Clean profiles\n")
    
    profile = input(
        "Profile (realistic/clean/mixed) [realistic]: "
    ).strip().lower() or "realistic"
    
    if profile not in PROFILES:
        print(f"Invalid profile '{profile}'. Using 'realistic'.")
        profile = "realistic"
    
    print(f"\nGenerating with profile: {profile}\n")
    
    for size in DATASET_SIZES:
        print(f"=== Dataset {size} ===")
        try:
            batch_size = 20 if size >= 100 else 10
            entities = generate_dataset(size, profile, batch_size)
            save_dataset(entities, size, profile)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except EntityGenerationError as e:
            print(f"Generation error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()