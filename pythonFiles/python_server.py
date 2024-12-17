from flask import Flask, request, jsonify
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import csv
import re
import pandas as pd

app = Flask(__name__)

# Load model and tokenizer
model_path = "model_state.pth"
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Helper function to remove emojis
def remove_emojis(text):
    emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002700-\U000027BF"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        "\U0000200D"             # zero-width joiner
        "\U00002328-\U0000232A"  # misc technical
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

# Scraping function to get reviews from an Amazon page
def scrape_amazon_reviews(url, output_csv="reviews.csv"):
    try:
        # Send a GET request to the provided URL
        response = requests.get(url)

        # Check for a successful response
        if response.status_code != 200:
            print(f"Error: Unable to fetch data, HTTP Status Code: {response.status_code}")
            return

        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all review elements
        review_elements = soup.select('[data-hook="review"]')

        # Initialize lists to store extracted data
        review_texts_list = []
        review_ratings_list = []

        # Loop through each review element
        for review_element in review_elements:
            # Extract the rating
            rating_element = review_element.select_one('.review-rating .a-icon-alt')
            if rating_element:
                rating_text = rating_element.text.strip().split()[0]
                review_ratings_list.append(float(rating_text))  # Convert to float
            else:
                review_ratings_list.append(None)

            # Extract the review text
            review_text_element = review_element.select_one('.review-text-content span')
            if review_text_element:
                review_text = review_text_element.text.strip()
                review_text = remove_emojis(review_text)  # Remove emojis
                review_texts_list.append(review_text)
            else:
                review_texts_list.append("N/A")

        # Write the data to a CSV file
        with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Review Text", "Star Rating"])

            for i in range(len(review_texts_list)):
                writer.writerow([review_texts_list[i], review_ratings_list[i]])

        print(f"Data successfully exported to {output_csv}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Predict whether review is real or fake
def get_prediction(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return "real" if predicted_class == 0 else "fake"

@app.route('/')
def home():
    return "Model backend is running!"

@app.route('/scrape_reviews', methods=['POST'])
def scrape_reviews():
    try:
        data = request.json
        url = data.get('url', '')
        if not url:
            return jsonify({"error": "URL is required"}), 400

        # Call the scrape function
        scrape_amazon_reviews(url, "reviews.csv")

        return jsonify({"message": "Scraping and saving reviews to CSV completed!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read reviews from the CSV
        df = pd.read_csv('reviews.csv')

        # Apply the prediction function to each review and add it to a new column
        df['Prediction'] = df['Review Text'].apply(get_prediction)

        # Save the updated CSV with predictions
        df.to_csv('reviews_with_predictions.csv', index=False)

        return jsonify({"message": "Predictions added successfully to reviews_with_predictions.csv"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_single', methods=['POST'])
def predict_single():
    try:
        data = request.json
        review_text = data.get('review_text', '')
        if not review_text:
            return jsonify({"error": "Review text is required"}), 400
        
        # Get the prediction for the single review text
        prediction = get_prediction(review_text)
        
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

