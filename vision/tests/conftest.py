import sys
from pathlib import Path

VISION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VISION))
sys.path.insert(0, str(Path(__file__).resolve().parent))
