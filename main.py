from flask import Flask, request, jsonify
from translator import Translator
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Initialize the Translator
translator = Translator()

@app.route("/translate", methods=["POST"])
def translate():
    data = request.json
    text = data.get("text", "")
    target_language = data.get("target_language", "")

    if not text or not target_language:
        return jsonify({"error": "Both 'text' and 'target_language' are required."}), 400

    try:
        translated_text = translator.translate_text(text, target_language)
        return jsonify({"translated_text": translated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)