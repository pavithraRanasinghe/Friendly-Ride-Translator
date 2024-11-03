from flask import Flask, request, jsonify
from transformers import pipeline
import torch
import langid

app = Flask(__name__)

# Load the language detection and translation models
lang_detector = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-960h")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-{target_language}")


def detect_language(text):
    """
    Detects the language of a given text.

    Args:
        text (str): The text to detect the language of.

    Returns:
        str: The ISO 639-1 language code (e.g., 'en' for English).
    """
    lang_code, _ = langid.classify(text)
    return lang_code

@app.route("/detect_and_translate", methods=["POST"])
def detect_and_translate():
    data = request.get_json()
    audio_data = data['audio']  # assuming audio is sent as base64 encoded string
    target_language = data['target_language']

    # Convert base64 audio to text
    text = lang_detector(audio_data)["text"]
    
    # Detect language - you may need a custom solution for accurate language detection
    detected_language = detect_language(text)  # placeholder function for language detection

    # Translate to target language if detected language and target are different
    if detected_language != target_language:
        translator.model = f"Helsinki-NLP/opus-mt-{detected_language}-{target_language}"
        translation = translator(text, max_length=400)
    else:
        translation = text

    return jsonify({"detected_language": detected_language, "translation": translation})

if __name__ == "__main__":
    app.run(debug=True)
