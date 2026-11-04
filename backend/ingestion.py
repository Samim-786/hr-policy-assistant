
from pathlib import Path

ALLOWED_EXTENSIONS = {".md", ".txt"}


def extract_text(file_path: Path) -> str:
    """
    Read text from a Markdown or TXT policy document.
    """

    # Validate file extension
    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Only .md and .txt files are supported.")

    try:
        # Read file using UTF-8 encoding
        text = file_path.read_text(encoding="utf-8")

    except UnicodeDecodeError as exc:
        raise ValueError("Unable to decode file. Please use UTF-8 encoding.") from exc

    except OSError as exc:
        raise ValueError("Unable to read the uploaded file.") from exc

    # Reject empty documents
    if not text.strip():
        raise ValueError("The uploaded file is empty.")

    return text
