"""Load the single authoritative, sourced Q1 model catalog."""
import json
from pathlib import Path
def updated_catalog():
    return json.loads((Path(__file__).resolve().parent/'research/model_catalog.json').read_text(encoding='utf-8'))
