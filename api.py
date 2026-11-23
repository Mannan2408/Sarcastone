"""
SarcasTone — Flask REST API
==============================
Backend API endpoints for the web demo.

Endpoints:
  POST /api/analyze   — Upload audio file, get sarcasm analysis
  GET  /api/health    — Health check
"""

import os
import tempfile
import numpy as np
import soundfile as sf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from src.config import FLASK_HOST, FLASK_PORT, STATIC_DIR, SAMPLE_RATE
from src.predictor import get_predictor


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")
    CORS(app)

    # Lazy-load predictor on first request
    predictor = None

    def _get_predictor():
        nonlocal predictor
        if predictor is None:
            predictor = get_predictor()
        return predictor

    @app.route("/")
    def index():
        """Serve the main frontend page."""
        return send_from_directory(STATIC_DIR, "index.html")

    @app.route("/api/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({"status": "ok", "service": "SarcasTone"})

    @app.route("/api/analyze", methods=["POST"])
    def analyze():
        """
        Analyze uploaded audio for sarcasm.

        Expects: multipart/form-data with 'audio' file field.
        Optional: 'text' field to override transcription.

        Returns: JSON with transcription, prediction, model comparison, audio analysis.
        """
        try:
            pred = _get_predictor()

            # Check for audio file
            if "audio" not in request.files:
                return jsonify({"error": "No audio file provided"}), 400

            audio_file = request.files["audio"]
            text_override = request.form.get("text", None)

            # Save to persistent temp cache for re-evaluation
            last_audio_path = os.path.join(tempfile.gettempdir(), "sarcastone_last_recording.wav")
            audio_file.save(last_audio_path)

            # Run prediction
            result = pred.predict(audio_path=last_audio_path, text_override=text_override)
            result["audio_url"] = "/api/audio/last"
            return jsonify(result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route("/api/reevaluate", methods=["POST"])
    def reevaluate():
        """Re-run analysis with edited text/emojis on the last audio recording."""
        try:
            data = request.get_json() or {}
            text = data.get("text", "").strip()
            if not text:
                return jsonify({"error": "No text provided"}), 400

            last_audio_path = os.path.join(tempfile.gettempdir(), "sarcastone_last_recording.wav")
            audio_url = "/api/audio/last"
            if not os.path.exists(last_audio_path):
                from src.config import DATA_DIR
                last_audio_path = os.path.join(DATA_DIR, "sample_000.wav")
                audio_url = "/api/audio/sample/0"

            pred = _get_predictor()
            result = pred.predict(audio_path=last_audio_path, text_override=text)
            result["audio_url"] = audio_url
            return jsonify(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route("/api/sample/<int:sample_id>", methods=["GET"])
    def analyze_sample(sample_id):
        """Analyze a preset sample without recording."""
        try:
            from src.config import DATA_DIR
            import json
            manifest_path = os.path.join(DATA_DIR, "manifest.json")
            if not os.path.exists(manifest_path):
                return jsonify({"error": "No samples found"}), 404
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            if sample_id < 0 or sample_id >= len(manifest):
                return jsonify({"error": "Sample ID out of range"}), 404
            sample = manifest[sample_id]
            wav_path = os.path.join(DATA_DIR, sample["filename"])
            pred = _get_predictor()
            result = pred.predict(audio_path=wav_path, text_override=sample["text"])
            result["audio_url"] = f"/api/audio/sample/{sample_id}"
            return jsonify(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route("/api/audio/last", methods=["GET"])
    def get_last_audio():
        """Serve the most recent user recording or uploaded audio snippet."""
        last_audio_path = os.path.join(tempfile.gettempdir(), "sarcastone_last_recording.wav")
        if os.path.exists(last_audio_path):
            return send_from_directory(tempfile.gettempdir(), "sarcastone_last_recording.wav", mimetype="audio/wav")
        from src.config import DATA_DIR
        fallback = os.path.join(DATA_DIR, "sample_000.wav")
        if os.path.exists(fallback):
            return send_from_directory(DATA_DIR, "sample_000.wav", mimetype="audio/wav")
        return jsonify({"error": "No audio file available"}), 404

    @app.route("/api/audio/sample/<int:sample_id>", methods=["GET"])
    def get_sample_audio(sample_id):
        """Serve audio file for a given benchmark sample."""
        try:
            from src.config import DATA_DIR
            import json
            manifest_path = os.path.join(DATA_DIR, "manifest.json")
            if not os.path.exists(manifest_path):
                return jsonify({"error": "No samples found"}), 404
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            if sample_id < 0 or sample_id >= len(manifest):
                return jsonify({"error": "Sample ID out of range"}), 404
            sample = manifest[sample_id]
            return send_from_directory(DATA_DIR, sample["filename"], mimetype="audio/wav")
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app
