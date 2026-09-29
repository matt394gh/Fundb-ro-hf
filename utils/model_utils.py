import json
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image, ImageOps

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
    return {"confidence_threshold": 0.35}


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
    1. Nutzt 'model/best.pt', falls lokal vorhanden.
    2. Versucht ein öffentliches Kleidungs-Modell von Hugging Face zu laden.
    3. Nutzt als zuverlässigen Fallback 'yolov8n.pt'.
    """
    from ultralytics import YOLO

    # 1. Lokale Modelldatei prüfen
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1000000:
        return YOLO(str(MODEL_PATH))

    # 2. Öffentliches HF-Modell laden (ohne Auth-Zwang)
    try:
        from huggingface_hub import hf_hub_download
        model_file = hf_hub_download(
            repo_id="BraveA/yolov8n-clothing",
            filename="best.pt"
        )
        return YOLO(model_file)
    except Exception:
        # Falls HF blockiert, geräuschlos auf lokales/Standard YOLO umschalten
        pass

    # 3. Standard-Modell von Ultralytics laden (lädt automatisch und ohne Fehler)
    return YOLO("yolov8n.pt")


def predict_clothing(image_file) -> dict:
    """Führt die Bildanalyse mit optimaler Bildaufbereitung durch."""
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
        # 1. Bild öffnen & RGB erzwingen
        img = Image.open(image_file).convert("RGB")

        # 2. Quadratisch ohne Verzerrung aufbereiten (640x640 mit weißem Rand)
        img_padded = ImageOps.pad(img, (640, 640), color=(255, 255, 255))

        # 3. Vorhersage ausführen
        results = model(img_padded)
        result = results[0]

        # 4. Auswertung für Klassifikationsmodell (probs vorhanden)
        if hasattr(result, "probs") and result.probs is not None:
            top_class_id = int(result.probs.top1)
            confidence = float(result.probs.top1conf)

            if top_class_id in custom_labels:
                label_name = custom_labels[top_class_id]
            elif hasattr(model, "names") and top_class_id in model.names:
                label_name = model.names[top_class_id]
            else:
                label_name = f"Klasse_{top_class_id}"

        # 5. Auswertung für Objekterkennungsmodell (boxes vorhanden)
        elif len(result.boxes) > 0:
            best_box = max(result.boxes, key=lambda b: float(b.conf[0]))
            class_id = int(best_box.cls[0])
            confidence = float(best_box.conf[0])

            if class_id in custom_labels:
                label_name = custom_labels[class_id]
            elif hasattr(model, "names") and class_id in model.names:
                label_name = model.names[class_id]
            else:
                label_name = f"Klasse_{class_id}"

        else:
            return {
                "label": "Sonstiges / Nicht erkannt",
                "confidence": 0.0,
                "probabilities": {}
            }

        # 6. Schwellenwert prüfen (Standard: 0.35)
        threshold = config.get("confidence_threshold", 0.35)
        if confidence < threshold:
            label_name = "Unbekannt / Nicht eindeutig"

        return {
            "label": label_name,
            "confidence": confidence,
            "probabilities": {label_name: confidence}
        }

    except Exception as e:
        st.error(f"Fehler bei der Bildanalyse: {e}")
        return {
            "label": "Fehler bei Analyse",
            "confidence": 0.0,
            "probabilities": {}
        }


def load_labels_list() -> list:
    """Kompatibilitätsfunktion für Auswahllisten im Upload-Formular."""
    labels_dict = load_labels()
    if labels_dict:
        return [labels_dict[k] for k in sorted(labels_dict.keys())]
    return ["T-Shirt", "Pullover", "Hose", "Schuhe", "Jacke", "Sonstiges"]


def get_model_info() -> dict:
    """Statusinfos für das Admin-Dashboard."""
    model = load_yolo_model()
    config = load_config()
    labels = load_labels()

    return {
        "model_loaded": model is not None,
        "uses_custom_file": MODEL_PATH.exists(),
        "labels_count": len(labels),
        "config": config
    }
