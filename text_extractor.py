import io
import os
import re
import uuid

from flask import current_app, jsonify
from werkzeug.utils import secure_filename


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"}


def allowed_file(filename: str) -> bool:
    if not filename or "." not in filename:
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def extract_text_from_bytes(file_bytes: bytes, ext: str) -> str:
    """
    Extract raw text from supported document types.
    """
    ext = ext.lower()
    if ext == ".pdf":
        from PyPDF2 import PdfReader

        pdf = PdfReader(io.BytesIO(file_bytes))
        chunks = []
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                chunks.append(page_text)
        return "\n".join(chunks).strip()

    if ext == ".docx":
        import docx

        doc = docx.Document(io.BytesIO(file_bytes))
        chunks = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(chunks).strip()

    if ext == ".txt":
        # Assume utf-8 first; tolerate legacy encodings.
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                return file_bytes.decode(encoding, errors="ignore").strip()
            except Exception:
                continue
        return file_bytes.decode(errors="ignore").strip()

    if ext in {".png", ".jpg", ".jpeg"}:
        from PIL import Image
        import pytesseract

        img = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(img) or ""
        return text.strip()

    raise ValueError(f"Unsupported file extension: {ext}")


def normalize_extracted_text(text: str) -> str:
    """
    Light normalization; heavy preprocessing happens later in ml_pipeline.
    """
    text = text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def save_uploaded_file(file_storage, upload_dir: str) -> str:
    """
    Save an uploaded file to disk using a safe randomized name.
    Returns the saved absolute path.
    """
    os.makedirs(upload_dir, exist_ok=True)
    original_name = secure_filename(file_storage.filename)
    ext = os.path.splitext(original_name)[1].lower()
    filename = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(upload_dir, filename)
    file_storage.save(path)
    return path


def extract_text_from_upload(file_storage) -> str:
    """
    Read uploaded file bytes and extract raw text.
    """
    if file_storage is None:
        raise ValueError("No file provided")

    filename = file_storage.filename or ""
    if not allowed_file(filename):
        raise ValueError("Unsupported file type")

    ext = os.path.splitext(filename)[1].lower()
    file_bytes = file_storage.read()
    text = extract_text_from_bytes(file_bytes, ext=ext)
    return normalize_extracted_text(text)


def validate_upload(file_storage, max_bytes: int) -> None:
    if file_storage is None:
        raise ValueError("No file provided")
    filename = file_storage.filename or ""
    if not allowed_file(filename):
        raise ValueError("Unsupported file type")
    # file_storage.content_length sometimes missing; if missing we can't validate size perfectly.
    content_length = getattr(file_storage, "content_length", None)
    if content_length is not None and content_length > max_bytes:
        raise ValueError(f"File too large. Max allowed is {max_bytes} bytes")

