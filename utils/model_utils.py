import json
from pathlib import Path
import streamlit as st
from PIL import Image, ImageOps
import torch
from transformers import CLIPProcessor, CLIPModel

# Pfade definieren
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
CONFIG_PATH = MODEL_DIR / "model_config.json"
LABELS_PATH = MODEL_DIR / "labels.txt"


def load_config() -> dict:
    """Lädt die Konfiguration aus model_config.json."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"confidence_threshold": 0.20}


def load_labels() -> list:
    """Lädt die Text-Labels aus labels.txt als Liste."""
    default_labels = ["T-Shirt", "Pullover", "Hoodie", "Jacke", "Hose", "Jeans", "Schuhe", "Tasche"]
    if not LABELS_PATH.exists():
        return default_labels

    labels = []
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels.append(parts[1].strip())
                else:
                    labels.append(line)
        return labels if labels else default_labels
    except Exception:
        return default_labels


@st.cache_resource
def load_clip_model():
    """Lädt das CLIP-Modell und den Processor von Hugging Face."""
    try:
        model_id = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
        return model, processor
    except Exception as e:
        st.error(f"Fehler beim Laden des CLIP-Modells: {e}")
        return None, None


def predict_clothing(image_file) -> dict:
    """Zero-Shot Klassifikation mit CLIP."""
    model, processor = load_clip_model()
    config = load_config()
    candidate_labels = load_labels()

    if model is None or processor is None:
        return {
            "label": "Fehler (Modell nicht geladen)",
            "confidence": 0.0,
            "probabilities": {}
        }

    try:
        # Bild öffnen und aufbereiten
        img = Image.open(image_file).convert("RGB")
        img_padded = ImageOps.pad(img, (224, 224), color=(255, 255, 255))

        # Prompts für CLIP formulieren (z. B. "a photo of a T-Shirt")
        text_prompts = [f"a photo of a {label}" for label in candidate_labels]

        # Eingaben für das Modell vorbereiten
        inputs = processor(
            text=text_prompts,
            images=img_padded,
            return_tensors="pt",
            padding=True
        )

        # Inferenz durchführen
        with torch.no_grad():
            outputs = model(**inputs)
            # Softmax über die Logits zur Ermittlung der Wahrscheinlichkeiten
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).squeeze().tolist()

        # Das wahrscheinlichste Label ermitteln
        if isinstance(probs, float):  # Falls nur ein Label existiert
            probs = [probs]

        best_idx = int(torch.argmax(torch.tensor(probs)))
        confidence = float(probs[best_idx])
        top_label = candidate_labels[best_idx]

        # Übersicht aller Wahrscheinlichkeiten erstellen
        probs_dict = {
            candidate_labels[i]: round(probs[i], 3)
            for i in range(len(candidate_labels))
        }

        threshold = config.get("confidence_threshold", 0.20)
        final_label = top_label if confidence >= threshold else "Unbekannt / Nicht eindeutig"

        return {
            "label": final_label,
            "confidence": confidence,
            "probabilities": probs_dict
        }

    except Exception as e:
        st.error(f"Fehler bei der CLIP-Analyse: {e}")
        return {
            "label": "Fehler bei Analyse",
            "confidence": 0.0,
            "probabilities": {}
        }


def load_labels_list() -> list:
    """Gibt die Liste der Labels für UI-Auswahllisten zurück."""
    return load_labels()


def get_model_info() -> dict:
    """Statusinformationen für das Admin-Dashboard."""
    model, _ = load_clip_model()
    labels = load_labels()
    config = load_config()

    return {
        "model_loaded": model is not None,
        "model_type": "CLIP Zero-Shot (openai/clip-vit-base-patch32)",
        "labels_count": len(labels),
        "labels": labels,
        "config": config
    }
