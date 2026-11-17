"""Load the YAML settings in vision/config/ and the product list."""
from pathlib import Path

import yaml

from . import VISION_DIR

CONFIG_FILE = VISION_DIR / "config" / "vision_config.yaml"


def resolve(path):
    """Paths in the config files are relative to the vision/ folder."""
    p = Path(path)
    return p if p.is_absolute() else VISION_DIR / p


def load_yaml(path):
    with open(resolve(path), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path=CONFIG_FILE):
    return load_yaml(path)


def load_products(path=VISION_DIR / "products.txt"):
    """Product names in order. Their order = the class numbers in the model."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def load_shelf_slots(path):
    """Returns {marker_id: {"name", "product", "item_offset_mm"}}."""
    data = load_yaml(path).get("slots", {})
    slots = {}
    for marker_id, slot in data.items():
        off = slot.get("item_offset_mm", [0, 0, 0])
        if len(off) != 3:
            raise ValueError(f"slot {marker_id}: item_offset_mm needs 3 numbers")
        slots[int(marker_id)] = {"name": slot.get("name", f"slot_{marker_id}"),
                                 "product": slot["product"],
                                 "item_offset_mm": [float(v) for v in off]}
    return slots
