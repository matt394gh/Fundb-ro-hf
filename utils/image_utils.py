import uuid
from pathlib import Path
from PIL import Image

UPLOAD_DIR = Path("uploads")


def save_uploaded_image(uploaded_file) -> str:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(uploaded_file.name).suffix or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    file_path = UPLOAD_DIR / filename

    img = Image.open(uploaded_file)
    img.save(file_path)
    return str(file_path)
