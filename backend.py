from flask import Flask, request, jsonify
from newspaper import Article
from textblob import TextBlob
import nltk
from imageToText.app import imageToText
from transformers import pipeline
from flask_cors import CORS


import cv2
import pytesseract

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # Update this path if necessary

# Summarization model
summarizer = pipeline('summarization', model='facebook/bart-large-cnn')

# Download the Punkt tokenizer
nltk.download('punkt')

app = Flask(__name__)
CORS(app)  # Allow all origins

@app.route('/analyze', methods=['POST'])
def analyze_article():
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    blob = TextBlob(article.text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    if polarity > 0:
        sentiment = 'positive'
    elif polarity < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    article.summary = article.summary.replace('\n', ' ').replace('\r', '')
    length = len(article.summary.split(" "))

    
    return jsonify({
        'title': article.title,
        'authors': article.authors,
        'publish_date': article.publish_date.strftime('%Y-%m-%d') if article.publish_date else None,
        'summary': article.summary,
        'polarity': polarity,
        'subjectivity': subjectivity,
        'sentiment': sentiment,
        'length': length
    })

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    data = request.get_json()
    image_url = data.get('image_url')
    
    if not image_url:
        return jsonify({'error': 'Image URL is required'}), 400
    
    try:
        text = imageToText(image_url)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Text summerization code here
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
    
    return jsonify({
        'text': summary 
    })

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data['text']
    if not text:
        return jsonify({'summary': 'No text provided.'}), 400
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']

    length = len(summary.split(" "))
    print(summary)
    return jsonify({'summary': summary, 'length': length})


if __name__ == '__main__':
    app.run(debug=True)
