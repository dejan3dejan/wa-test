import os
import sys
import json
import time
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel
from tqdm import tqdm
from dotenv import load_dotenv

from google import genai
from google.genai import types

# Add logging
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 1. Setup & Config
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.critical("GEMINI_API_KEY not found in .env")
    sys.exit(1)

# 2. Define Structured Output Schema
class AuditResult(BaseModel):
    label: Literal["factual", "hallucinated"]
    reasoning: str
    confidence_score: float

# 3. Define the Judge (Model)
def get_audit_model():
    """Initialize Gemini client with structured output configuration."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    system_instruction = """You are an expert Industrial Engineer and Data Auditor. 
Your task is to verify if a Description matches the Object Name.
- If the Description plausibly describes the Object Name, label 'factual'.
- If the Description describes a COMPLETELY DIFFERENT object, contradicts the name (e.g., Temperature Sensor described as measuring pressure), or is nonsensical, label 'hallucinated'.
- Be strict about contradictions.
"""
    
    config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=AuditResult,
        system_instruction=system_instruction
    )
    
    return client, config

def audit_entity(client, config, name: str, description: str, type_name: str = "") -> AuditResult:
    """Audit a single entity using Gemini."""
    prompt = f"""
Object Name: {name}
Type: {type_name}
Description: {description}

Analyze accuracy.
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config
        )
        return AuditResult.model_validate_json(response.text)
    except Exception as e:
        # Fallback for API errors
        logger.error(f"Error auditing '{name}': {e}")
        return AuditResult(label="hallucinated", reasoning=f"API Error: {str(e)}", confidence_score=0.0)

def run_audit(input_file: str, limit: int = None):
    input_path = Path(input_file)
    if not input_path.exists():
        logger.error(f"File not found: {input_path}")
        return

    logger.info(f"Audit Start: {input_path.name}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if limit:
        data = data[:limit]
        logger.info(f"Limiting to first {limit} entries.")

    client, config = get_audit_model()
    audited_data = []
    
    factual_count = 0
    hallucinated_count = 0

    for entity in tqdm(data, desc="Auditing"):
        name = entity.get('name', 'Unknown')
        desc = entity.get('description', '')
        type_name = entity.get('type_name', '')
        
        # Skip if already audited (optional check)
        # if 'audit_label' in entity: ...

        result = audit_entity(client, config, name, desc, type_name)
        
        # Enrich entity
        entity['audit_label'] = result.label
        entity['audit_reasoning'] = result.reasoning
        entity['audit_confidence'] = result.confidence_score
        
        audited_data.append(entity)
        
        if result.label == "factual":
            factual_count += 1
        else:
            hallucinated_count += 1
            
    # Save Results
    output_filename = input_path.stem + "_audited" + input_path.suffix
    output_path = input_path.parent / output_filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(audited_data, f, indent=2)
        
    logger.info(f"{'-'*40}")
    logger.info(f"Audit Complete!")
    logger.info(f"Total: {len(audited_data)}")
    logger.info(f"Factual: {factual_count} ({(factual_count/len(audited_data)):.1%})")
    logger.info(f"Hallucinations: {hallucinated_count} ({(hallucinated_count/len(audited_data)):.1%})")
    logger.info(f"Saved to: {output_path}")
    logger.info(f"{'-'*40}\n")
    
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.info("Usage: python scripts/audit_generator.py <path_to_json_file> [limit]")
        logger.info("Example: python scripts/audit_generator.py datasets/golden_descriptions_10.json")
    else:
        file_path = sys.argv[1]
        limit_val = int(sys.argv[2]) if len(sys.argv) > 2 else None
        
        # Adjust path if relative to project root
        if not os.path.isabs(file_path):
            file_path = os.path.join(project_root, file_path)
            
        run_audit(file_path, limit_val)
