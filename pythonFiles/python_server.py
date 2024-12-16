from flask import Flask, request, jsonify
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

app = Flask(__name__)
model_path = "model_state.pth"
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) 
model.eval() 

@app.route('/')
def home():
    return "Model backend is running!"
@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.json
        text = data.get('text', '')

        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

        return jsonify({"predicted_class": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
