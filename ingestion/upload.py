from pathlib import Path

ORIGINAL_DIR = Path("storage/original")


def save_upload(filename: str, data: bytes) -> Path:
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    destination = ORIGINAL_DIR / Path(filename).name
    destination.write_bytes(data)
    return destination
