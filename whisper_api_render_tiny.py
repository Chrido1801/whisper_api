from flask import Flask, request, jsonify
import whisper
import requests
import tempfile
import os

app = Flask(__name__)
model = whisper.load_model("tiny")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        data = request.get_json()
        url = data.get("url")
        if not url:
            return jsonify({"error": "Missing 'url' in request body"}), 400

        # Download audio file
        response = requests.get(url)
        if response.status_code != 200:
            return jsonify({"error": f"Failed to download file: {response.status_code}"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # Transcribe
        result = model.transcribe(tmp_path)
        os.remove(tmp_path)

        return jsonify({
            "text": result["text"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Whisper API is running!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0")
