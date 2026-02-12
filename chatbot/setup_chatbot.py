"""
Chatbot Project Bootstrap Script
Creates folder structure and placeholder files.
Safe to re-run (won't overwrite existing files).
"""

from pathlib import Path

# Root folder (current directory)
ROOT = Path(__file__).parent

FOLDERS = [
    ROOT / "templates",
    ROOT / "static",
    ROOT / "data"
]

FILES = {
    ROOT / "app.py": """# FastAPI entry point\n\napp = None  # TODO: Implement FastAPI app\n""",

    ROOT / "model_loader.py": """# Model loading logic\n\nclass ModelManager:\n    pass\n""",

    ROOT / "feedback_store.py": """# Feedback saving logic\n\ndef save_feedback(data: dict):\n    pass\n""",

    ROOT / "requirements.txt": """fastapi\nuvicorn\ntorch\ntransformers\njinja2\npyyaml\n""",

    ROOT / "templates" / "index.html": """<!DOCTYPE html>\n<html>\n<head>\n    <title>EMTAC Chat</title>\n</head>\n<body>\n    <h2>Chatbot Placeholder</h2>\n</body>\n</html>\n""",

    ROOT / "static" / "style.css": """body { font-family: Arial; }\n""",

    ROOT / "data" / "feedback_log.jsonl": ""
}


def create_folders():
    for folder in FOLDERS:
        if not folder.exists():
            folder.mkdir(parents=True)
            print(f"[CREATED] Folder: {folder}")
        else:
            print(f"[EXISTS] Folder: {folder}")


def create_files():
    for path, content in FILES.items():
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            print(f"[CREATED] File: {path}")
        else:
            print(f"[EXISTS] File: {path}")


def main():
    print("Bootstrapping Chatbot Project...\n")
    create_folders()
    create_files()
    print("\nSetup Complete.")


if __name__ == "__main__":
    main()
