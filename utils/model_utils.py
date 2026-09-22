import streamlit as st
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Hugging Face Repository
HF_REPO_ID = "kesimeg/yolov8n-clothing-detection"
MODEL_FILENAME = "best.pt"


@st.cache_resource
def load_yolo_model():
    """Lädt das YOLO-Modell von Hugging Face mit Caching."""
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
        model = YOLO(model_path)
        return model
    except Exception:
        try:
            model = YOLO(f"hf://{HF_REPO_ID}")
            return model
        except Exception as e:
            st.error(f"Fehler beim Laden des Hugging Face Modells: {e}")
            return None


def predict_clothing(image_file) -> dict:
    """Analyse des Kleidungsstücks mit dem Hugging Face YOLO-Modell."""
    model = load_yolo_model()

    if model is None:
        return {
            "label": "Sonstiges (Modell nicht geladen)",
            "confidence": 0.0,
            "probabilities": {}
        }

    try:
        img = Image.open(image_file).convert("RGB")
        results = model(img)
        result = results[0]

        # Auswertung für Klassifikations-Modelle
        if hasattr(result, "probs") and result.probs is not None:
            probs = result.probs.data.cpu().numpy()
            top_class_id = int(result.probs.top1)
            top_conf = float(result.probs.top1conf)
            top_label = model.names[top_class_id]

            top3_indices = probs.argsort()[-3:][::-1]
            prob_dict = {model.names[i]: float(probs[i]) for i in top3_indices}

            return {
                "label": top_label,
                "confidence": top_conf,
                "probabilities": prob_dict
            }

        # Auswertung für Objekterkennungs-Modelle (Detection)
        elif hasattr(result, "boxes") and len(result.boxes) > 0:
            best_box = max(result.boxes, key=lambda b: float(b.conf[0]))
            class_id = int(best_box.cls[0])
            confidence = float(best_box.conf[0])
            label_name = model.names[class_id]

            return {
                "label": label_name,
                "confidence": confidence,
                "probabilities": {label_name: confidence}
            }

        else:
            return {
                "label": "Keine Bekleidung erkannt",
                "confidence": 0.0,
                "probabilities": {}
            }

    except Exception as e:
        st.error(f"Fehler bei der KI-Analyse: {e}")
        return {
            "label": "Fehler bei Analyse",
            "confidence": 0.0,
            "probabilities": {}
        }


def load_labels() -> list:
    """Liest die verfügbaren Kategorien aus dem Modell aus."""
    model = load_yolo_model()
    if model and hasattr(model, "names"):
        return list(model.names.values())
    return ["Pullover", "T-Shirt", "Hoodie", "Jacke", "Hose", "Schuhe", "Sonstiges"]


def get_model_info() -> dict:
    """Informationen zum Hugging Face Modell."""
    model = load_yolo_model()
    return {
        "model_loaded": model is not None,
        "huggingface_repo": HF_REPO_ID,
        "model_type": "YOLOv8 (Hugging Face)",
        "detected_classes": load_labels()
    }
