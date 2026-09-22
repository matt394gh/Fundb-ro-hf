import os
import json
from pathlib import Path
from PIL import Image
import streamlit as st

# Pfade definieren (Hauptverzeichnis des Projekts)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best.pt"
CONFIG_PATH = BASE_DIR / "model" / "model_config.json"
LABELS_PATH = BASE_DIR / "model" / "labels.txt"

# Hugging Face Secrets aus Streamlit Cloud auslesen
HF_REPO_ID = st.secrets.get("HF_REPO_ID", None)  # z. B. "user/fundbuero-yolo"
HF_FILENAME = st.secrets.get("HF_FILENAME", "best.pt")


def load_config() -> dict:
    """Lädt die Konfigurationsdatei für das Modell."""
    default_config = {
        "confidence_threshold": 0.25,
        "model_type": "yolo"
    }
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                default_config.update(config)
        except Exception:
            pass
    return default_config


def load_labels() -> list:
    """Lädt und bereinigt die Labels aus labels.txt."""
    if not LABELS_PATH.exists():
        return ["T-Shirt", "Pullover", "Hoodie", "Jacke", "Hose", "Schuhe", "Sonstiges"]

    labels = []
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) > 1 and parts[0].replace(":", "").replace("-", "").isdigit():
                    clean_label = parts[1].strip()
                else:
                    clean_label = line
                labels.append(clean_label)

        return labels if labels else ["Sonstiges"]
    except Exception:
        return ["Sonstiges"]


@st.cache_resource
def load_yolo_model():
    """
    Lädt das YOLO-Modell (.pt) aus dem lokalen Ordner oder lädt es von Hugging Face herunter.
    """
    target_path = MODEL_PATH

    # 1. Download von Hugging Face, falls nicht lokal vorhanden
    if not target_path.exists() and HF_REPO_ID:
        try:
            from huggingface_hub import hf_hub_download
            hf_token = st.secrets.get("HF_TOKEN", None)
            
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
                token=hf_token
            )
            target_path = Path(downloaded_path)
        except Exception as e:
            st.warning(f"Hugging Face Download fehlgeschlagen: {e}")

    # 2. Prüfen, ob Modell existiert
    if not target_path.exists():
        return None

    # 3. YOLO Modell via Ultralytics laden
    try:
        from ultralytics import YOLO
        model = YOLO(str(target_path))
        return model
    except Exception as e:
        st.error(f"Fehler beim Laden des YOLO-Modells (`{target_path.name}`): {e}")
        return None


def predict_clothing(image_file) -> dict:
    """Führt die YOLO-Erkennung auf dem hochgeladenen Bild aus."""
    config = load_config()
    model = load_yolo_model()

    if model is None:
        return {
            "label": "Sonstiges (YOLO-Modell nicht geladen)",
            "confidence": 0.0,
            "probabilities": {}
        }

    try:
        # Bild öffnen
        img = Image.open(image_file).convert("RGB")

        # YOLO Inferenz ausführen
        results = model(img)
        result = results[0]

        # A) Falls es ein YOLO-Klassifikationsmodell ist (Cls)
        if hasattr(result, "probs") and result.probs is not None:
            top_class_id = int(result.probs.top1)
            top_conf = float(result.probs.top1conf)
            top_label = result.names[top_class_id]

            # Top 3 extrahieren
            top3_ids = result.probs.top5[:3] if hasattr(result.probs, "top5") else [top_class_id]
            probs_dict = {result.names[int(cid)]: float(result.probs.data[int(cid)]) for cid in top3_ids}

        # B) Falls es ein YOLO-Objekterkennungsmodell ist (Det)
        elif hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
            # Höchste Konfidenz unter den gefundenen Boxen wählen
            best_box = max(result.boxes, key=lambda b: float(b.conf[0]))
            top_class_id = int(best_box.cls[0])
            top_conf = float(best_box.conf[0])
            top_label = result.names[top_class_id]

            probs_dict = {top_label: top_conf}
        else:
            return {
                "label": "Keine Kleidung erkannt",
                "confidence": 0.0,
                "probabilities": {}
            }

        threshold = config.get("confidence_threshold", 0.25)
        final_label = top_label if top_conf >= threshold else "Unbekannt / Nicht eindeutig"

        return {
            "label": final_label,
            "confidence": top_conf,
            "probabilities": probs_dict
        }

    except Exception as e:
        st.error(f"Fehler bei der YOLO-Analyse: {e}")
        return {
            "label": "Fehler bei Analyse",
            "confidence": 0.0,
            "probabilities": {}
        }


def get_model_info() -> dict:
    """Informationen für das Admin-Dashboard."""
    model = load_yolo_model()
    labels = load_labels()
    config = load_config()

    file_exists = MODEL_PATH.exists()
    file_size_mb = round(MODEL_PATH.stat().st_size / (1024 * 1024), 2) if file_exists else 0

    return {
        "model_loaded": model is not None,
        "model_type": "YOLO (.pt)",
        "model_path": str(MODEL_PATH),
        "file_exists": file_exists,
        "file_size_mb": f"{file_size_mb} MB" if file_exists else "N/A",
        "labels_count": len(labels),
        "labels": labels,
        "config": config
    }
