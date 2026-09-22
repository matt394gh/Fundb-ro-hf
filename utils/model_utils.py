import os
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import streamlit as st

# Pfade definieren (Hauptverzeichnis des Projekts)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "keras_model.h5"
CONFIG_PATH = BASE_DIR / "model" / "model_config.json"
LABELS_PATH = BASE_DIR / "model" / "labels.txt"

# Falls Hugging Face verwendet werden soll, Repository hier anpassen
HF_REPO_ID = st.secrets.get("HF_REPO_ID", None)  # z. B. "user/fundbuero-model"
HF_FILENAME = st.secrets.get("HF_FILENAME", "model.h5")


def load_config() -> dict:
    """Lädt die Konfigurationsdatei für das Modell."""
    default_config = {
        "image_width": 224,
        "image_height": 224,
        "normalization": "0_1",
        "confidence_threshold": 0.55,
        "model_type": "keras_h5"
    }
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                default_config.update(config)
        except Exception as e:
            st.warning(f"Hinweis: Config konnte nicht geladen werden ({e}). Standardwerte werden genutzt.")
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
                # Zahlenpräfixe wie "0 T-Shirt" oder "1: Pullover" entfernen
                parts = line.split(" ", 1)
                if len(parts) > 1 and parts[0].replace(":", "").replace("-", "").isdigit():
                    clean_label = parts[1].strip()
                else:
                    clean_label = line
                labels.append(clean_label)

        return labels if labels else ["Sonstiges"]
    except Exception as e:
        st.error(f"Fehler beim Lesen von labels.txt: {e}")
        return ["Sonstiges"]


@st.cache_resource
def load_keras_model():
    """
    Lädt das Modell aus dem lokalen Pfad oder lädt es von Hugging Face herunter.
    Gibt None zurück, falls kein Modell verfügbar ist.
    """
    target_path = MODEL_PATH

    # 1. Prüfen, ob Download von Hugging Face gewünscht ist
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

    # 2. Prüfen, ob die Modelldatei lokal existiert
    if not target_path.exists():
        return None

    # 3. Keras / TensorFlow Modell laden
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(target_path), compile=False)
        return model
    except Exception as e:
        st.error(f"Fehler beim Laden des Keras-Modells (`{target_path.name}`): {e}")
        return None


def predict_clothing(image_file) -> dict:
    """Führt die Klassifizierung des hochgeladenen Bildes durch."""
    config = load_config()
    labels = load_labels()
    model = load_keras_model()

    # Fallback, wenn kein Modell geladen werden konnte
    if model is None:
        return {
            "label": "Sonstiges (KI-Modell nicht geladen)",
            "confidence": 0.0,
            "probabilities": {}
        }

    try:
        # Bild vorbereiten
        img = Image.open(image_file).convert("RGB")
        target_size = (config.get("image_width", 224), config.get("image_height", 224))
        img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)

        # In Numpy-Array umwandeln
        img_array = np.asarray(img, dtype=np.float32)

        # Normalisierung anwenden
        norm_type = config.get("normalization", "0_1")
        if norm_type == "0_1":
            img_array = img_array / 255.0
        elif norm_type == "minus1_1":
            img_array = (img_array / 127.5) - 1.0

        # Batch-Dimension hinzufügen
        img_array = np.expand_dims(img_array, axis=0)

        # Vorhersage
        preds = model.predict(img_array)[0]

        # Wahrscheinlichkeiten zuordnen
        probs = {}
        for idx, prob in enumerate(preds):
            lbl = labels[idx] if idx < len(labels) else f"Klasse_{idx}"
            probs[lbl] = float(prob)

        # Nach Wahrscheinlichkeit sortieren
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_label, top_conf = sorted_probs[0]

        threshold = config.get("confidence_threshold", 0.55)
        final_label = top_label if top_conf >= threshold else "Unbekannt / Nicht eindeutig"

        return {
            "label": final_label,
            "confidence": top_conf,
            "probabilities": dict(sorted_probs[:3])
        }

    except Exception as e:
        st.error(f"Fehler bei der KI-Bildanalyse: {e}")
        return {
            "label": "Fehler bei Analyse",
            "confidence": 0.0,
            "probabilities": {}
        }


def get_model_info() -> dict:
    """Gibt Statusinformationen für den Admin-Bereich zurück."""
    model = load_keras_model()
    labels = load_labels()
    config = load_config()

    file_exists = MODEL_PATH.exists()
    file_size_mb = round(MODEL_PATH.stat().st_size / (1024 * 1024), 2) if file_exists else 0

    return {
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "file_exists": file_exists,
        "file_size_mb": f"{file_size_mb} MB" if file_exists else "N/A",
        "labels_count": len(labels),
        "labels": labels,
        "config": config
    }
