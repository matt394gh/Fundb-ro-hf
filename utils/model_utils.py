import json
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image

# Pfade definieren (Hauptverzeichnis des Projekts & model/-Ordner)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "best.pt"
CONFIG_PATH = MODEL_DIR / "model_config.json"
LABELS_PATH = MODEL_DIR / "labels.txt"


def load_config() -> dict:
    """Lädt die Konfigurationsdatei aus dem model/-Ordner."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Fehler beim Laden von model_config.json: {e}")
    return {"confidence_threshold": 0.25}


def load_labels() -> dict:
    """
    Lädt die Label-Zuordnung aus labels.txt.
    Erwartetes Format: '0 T-Shirt' oder 'T-Shirt' (Zeile für Zeile).
    """
    if not LABELS_PATH.exists():
        return {}

    labels_map = {}
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels_map[int(parts[0])] = parts[1].strip()
                else:
                    labels_map[idx] = line
        return labels_map
    except Exception as e:
        st.warning(f"Fehler beim Lesen von labels.txt: {e}")
        return {}


@st.cache_resource
def load_yolo_model():
    """
    Lädt das YOLO-Modell.
    Nutzt 'model/best.pt', falls vorhanden.
    Lädt andernfalls automatisch 'yolov8n.pt' direkt aus dem Internet herunter.
    """
    try:
        from ultralytics import YOLO

        if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1000000:
            model_target = str(MODEL_PATH)
        else:
            # Automatischer Download des Standard-Modells
            model_target = "yolov8n.pt"

        model = YOLO(model_target)
        return model
    except Exception as e:
        st.error(f"❌ Fehler beim Laden des YOLO-Modells: {e}")
        return None


def predict_clothing(image_file) -> dict:
    """Führt die Bildanalyse mit dem YOLO-Modell durch."""
    model = load_yolo_model()
    config = load_config()
    custom_labels = load_labels()

    if model is None:
        return {
            "label": "Sonstiges (Modell nicht geladen)",
            "confidence": 0.0,
            "probabilities": {}
        }

    try:
        # Bild öffnen und Farbraum sichern
        img = Image.open(image_file).convert("RGB")

        # YOLO Vorhersage ausführen
        results = model(img)
        result = results[0]

        # Falls Bounding Boxes / Objekte erkannt wurden
        if len(result.boxes) > 0:
            # Box mit der höchsten Wahrscheinlichkeit auswählen
            best_box = max(result.boxes, key=lambda b: float(b.conf[0]))
            class_id = int(best_box.cls[0])
            confidence = float(best_box.conf[0])

            # Label aus labels.txt oder direkt aus dem Modell holen
            if class_id in custom_labels:
                label_name = custom_labels[class_id]
            elif hasattr(model, "names") and class_id in model.names:
                label_name = model.names[class_id]
            else:
                label_name = f"Klasse_{class_id}"

            # Schwellenwert prüfen
            threshold = config.get("confidence_threshold", 0.25)
            if confidence < threshold:
                label_name = "Unbekannt / Nicht eindeutig"

            return {
                "label": label_name,
                "confidence": confidence,
                "probabilities": {label_name: confidence}
            }
        else:
            return {
                "label": "Sonstiges / Nicht erkannt",
                "confidence": 0.0,
                "probabilities": {}
            }

    except Exception as e:
        st.error(f"Fehler bei der Bildanalyse: {e}")
        return {
            "label": "Fehler bei Analyse",
            "confidence": 0.0,
            "probabilities": {}
        }


def load_labels_list() -> list:
    """Kompatibilitätsfunktion für das Upload-Formular."""
    labels_dict = load_labels()
    if labels_dict:
        return [labels_dict[k] for k in sorted(labels_dict.keys())]
    return ["T-Shirt", "Pullover", "Hose", "Schuhe", "Sonstiges"]


def get_model_info() -> dict:
    """Gibt Infos über den Modellstatus für das Dashboard aus."""
    model = load_yolo_model()
    config = load_config()
    labels = load_labels()

    return {
        "model_loaded": model is not None,
        "uses_custom_file": MODEL_PATH.exists(),
        "labels_count": len(labels),
        "config": config
    }
