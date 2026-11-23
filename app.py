"""
SarcasTone — Main Entry Point
================================
Run this to start the web application.

Usage:
    python app.py

Then open http://localhost:5000 in your browser.
"""

import webbrowser
import threading
from src.api import create_app
from src.config import FLASK_HOST, FLASK_PORT


def open_browser():
    """Open the browser after a short delay to let the server start."""
    import time
    time.sleep(2)
    url = f"http://{FLASK_HOST}:{FLASK_PORT}"
    print(f"\n🌐 Opening browser at {url}")
    webbrowser.open(url)


def main():
    print("=" * 50)
    print("  🎭 SarcasTone — Sarcasm Detection Demo")
    print("=" * 50)
    print()
    print("  Starting server...")
    print(f"  URL: http://{FLASK_HOST}:{FLASK_PORT}")
    print()
    print("  Press Ctrl+C to stop")
    print()

    app = create_app()

    # Open browser automatically
    threading.Thread(target=open_browser, daemon=True).start()

    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)


if __name__ == "__main__":
    main()
