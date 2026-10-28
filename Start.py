"""One-step local launcher: python Start.py

Creates the virtual environment if needed, installs dependencies when
requirements.txt has changed, then starts the Streamlit app.
"""
import hashlib
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / "venv"
REQUIREMENTS = ROOT / "requirements.txt"
HASH_FILE = VENV_DIR / ".requirements.sha256"

if os.name == "nt":
    VENV_PYTHON = VENV_DIR / "Scripts" / "python.exe"
else:
    VENV_PYTHON = VENV_DIR / "bin" / "python"


def run(*args):
    subprocess.run([str(VENV_PYTHON), *args], cwd=ROOT, check=True)


def main():
    if not VENV_PYTHON.exists():
        print("Creating virtual environment...")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)

    # supabase is imported by the app but is not always pinned in requirements.txt
    wanted = REQUIREMENTS.read_bytes() if REQUIREMENTS.exists() else b""
    digest = hashlib.sha256(wanted + b"streamlit supabase").hexdigest()
    if not HASH_FILE.exists() or HASH_FILE.read_text().strip() != digest:
        print("Installing dependencies (first run can take a while)...")
        run("-m", "pip", "install", "--upgrade", "pip")
        run("-m", "pip", "install", "streamlit", "supabase")
        if REQUIREMENTS.exists():
            run("-m", "pip", "install", "-r", str(REQUIREMENTS))
        HASH_FILE.write_text(digest)

    print("Starting Streamlit...")
    try:
        run("-m", "streamlit", "run", "Home.py")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
