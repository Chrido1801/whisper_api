
from flask import Flask, request, jsonify
import whisper
import requests
import tempfile
import os

app = Flask(__name__)
model = whisper.load_model("base")  # Alternativen: "tiny", "base", "small", "medium", "large"

@app.route("/", methods=["GET"])
def home():
    return "✅ Whisper API ist bereit!", 200

@app.route("/whisper", methods=["POST"])
def transcribe():
    try:
        audio_url = request.json.get("url")
        if not audio_url:
            return jsonify({"error": "Kein Audio-Link übergeben"}), 400

        # MP3-Datei temporär herunterladen
        r = requests.get(audio_url)
        if r.status_code != 200:
            return jsonify({"error": "Audio konnte nicht geladen werden"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(r.content)
            temp_path = f.name

        # Whisper-Transkription
        result = model.transcribe(temp_path, language="de")
        os.remove(temp_path)
        return jsonify({"text": result["text"]}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
