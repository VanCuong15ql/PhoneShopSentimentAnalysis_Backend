from flask import Flask, request, jsonify
import torch
import os
from transformers import AutoTokenizer
from predict_aspectsV2 import ABSAModel, preprocess_fn, predict_sentiment

app = Flask(__name__)

# Load model and tokenizer once at startup
model_path = os.path.join(os.getcwd(), 'model', 'best_model.pt')
model = ABSAModel().to('cpu')
checkpoint = torch.load(model_path, map_location=torch.device('cpu'),weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

PHOBERT_NAME = "./phobert-base"
tokenizer = AutoTokenizer.from_pretrained(PHOBERT_NAME)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        predicted_aspects = predict_sentiment(text, model, tokenizer)
        if not predicted_aspects:
            return jsonify({'error': 'No aspects predicted'}), 400

        formatted_output = [f"{label}:{sentiment}" for label, sentiment in predicted_aspects]

        return jsonify({'aspects': formatted_output})
    except Exception as e:
        print(f"Error: {str(e)}")  # Ghi log chi tiết về lỗi
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5001)