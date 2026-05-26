"""
Start the Flask API from the backend folder.

Usage (from project/backend):
    python run.py
"""
import os
import sys

# project/ must be on sys.path so "backend" is importable as a package
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import app  # noqa: E402

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"Starting API on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
