# Spam Email Detector

A lightweight web app that classifies messages as SPAM or HAM with confidence percentage. Built using Naive Bayes and TF-IDF vectorization with a clean dark-themed Flask frontend.

## Features
- Detects spam messages with 98% accuracy
- Shows confidence percentage for every prediction
- Clean dark-themed web interface
- Real-time prediction on any input message

## Tech Stack
- Python
- Scikit-learn (Naive Bayes, TF-IDF)
- Flask
- HTML, CSS

## Model Performance
- Accuracy: 98.03%
- Precision: 0.99
- Recall: 0.98
- F1 Score: 0.98
- Dataset: SMS Spam Collection (5572 messages)

## How to Run
1. Clone the repository
2. Install dependencies: pip install flask scikit-learn pandas numpy
3. Run the model: python model.py
4. Run the app: python app.py
5. Open browser and go to http://127.0.0.1:5000
