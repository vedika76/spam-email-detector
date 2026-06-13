from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    message = None
    
    if request.method == 'POST':
        message = request.form['message']
        message_tfidf = tfidf.transform([message])
        pred = model.predict(message_tfidf)[0]
        prob = model.predict_proba(message_tfidf)[0]
        prediction = "SPAM" if pred == 1 else "HAM"
        confidence = round(max(prob) * 100, 2)
    
    return render_template('index.html', prediction=prediction, confidence=confidence, message=message)

if __name__ == '__main__':
    app.run(debug=True)