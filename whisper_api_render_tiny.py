from flask import Flask, request, jsonify
import requests
import tempfile
import os
import whisper
from pydub import AudioSegment

app = Flask(__name__)
model = whisper.load_model("tiny")  # oder "base", "small", etc.

def download_and_clean_audio(url):
    # Lade und speichere temporär
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise Exception(f"Download failed: {response.status_code}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        tmp_mp3_path = tmp_file.name

    # Optional: saubere Neu-Konvertierung mit pydub
    clean_path = tmp_mp3_path.replace(".mp3", "_clean.mp3")
    audio = AudioSegment.from_mp3(tmp_mp3_path)
    audio.export(clean_path, format="mp3")

    # Ursprungsdatei löschen
    os.remove(tmp_mp3_path)
    return clean_path

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        data = request.get_json()
        audio_url = data.get("url")
        if not audio_url:
            return jsonify({"error": "Missing 'url' in request"}), 400

        audio_path = download_and_clean_audio(audio_url)
        result = model.transcribe(audio_path)

        # Aufräumen
        os.remove(audio_path)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
