from transformers import MarianTokenizer, MarianMTModel

class Translator:
    def __init__(self):
        self.models = {}  # Cache for loaded models

    def load_model(self, target_language):
        # Construct the model name dynamically
        model_name = f"Helsinki-NLP/opus-mt-en-{target_language}"
        
        if model_name not in self.models:
            # Load tokenizer and model for the language pair
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            self.models[model_name] = (tokenizer, model)

        return self.models[model_name]

    def translate_text(self, text, target_language):
        # Load the appropriate model and tokenizer
        tokenizer, model = self.load_model(target_language)

        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # Generate translation
        translated = model.generate(**inputs)

        # Decode and return the translation
        return tokenizer.decode(translated[0], skip_special_tokens=True)
