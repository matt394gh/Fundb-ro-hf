import json
import uuid
from pathlib import Path

DATA_FILE = Path("data/items.json")


def _ensure_data_file():
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)


def load_items() -> list:
    _ensure_data_file()
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_item(item_data: dict) -> str:
    _ensure_data_file()
    items = load_items()
    item_id = f"ITEM-{uuid.uuid4().hex[:6].upper()}"
    item_data["id"] = item_id
    items.append(item_data)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return item_id


def update_item(item_id: str, updated_fields: dict) -> bool:
    items = load_items()
    for item in items:
        if item.get("id") == item_id:
            item.update(updated_fields)
            with open(DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            return True
    return False


def delete_item(item_id: str) -> bool:
    items = load_items()
    filtered = [i for i in items if i.get("id") != item_id]
    if len(filtered) < len(items):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        return True
    return False


def get_item_by_id(item_id: str):
    items = load_items()
    for item in items:
        if item.get("id") == item_id:
            return item
    return None


def filter_items(items, query=None, category=None, color=None, location=None, status=None):
    res = items
    if query:
        q = query.lower()
        res = [
            i for i in res
            if q in i.get("description", "").lower()
            or q in i.get("brand", "").lower()
            or q in i.get("id", "").lower()
        ]
    if category and category != "Alle":
        res = [i for i in res if i.get("category") == category]
    if color and color != "Alle":
        res = [i for i in res if i.get("color") == color]
    if location and location != "Alle":
        res = [i for i in res if i.get("location_found") == location]
    if status and status != "Alle":
        res = [i for i in res if i.get("status") == status]
    return res


def get_unique_categories(items):
    return sorted(list({i.get("category") for i in items if i.get("category")}))


def get_unique_colors(items):
    return sorted(list({i.get("color") for i in items if i.get("color")}))


def get_unique_locations(items):
    return sorted(list({i.get("location_found") for i in items if i.get("location_found")}))


def export_data_as_json():
    return json.dumps(load_items(), ensure_ascii=False, indent=2)
