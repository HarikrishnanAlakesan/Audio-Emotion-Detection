from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import speech_recognition as sr

app = Flask(__name__)

# List of bad words that may indicate negative sentiment
bad_words = ["Fuck", "Bitch", "Shit", "Idiot", "Fool"]  # Add your bad words here

def recognize_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        print('Reading audio file...')
        recorded_audio = recognizer.record(source)
        print('Done reading..')

    print('Transcribing the audio...')
    text = recognizer.recognize_google(recorded_audio, language='en-US')
    print('Transcription:', text)
    return text

def contains_bad_words(text):
    # Convert both bad words and text to lowercase for case insensitivity
    lower_text = text.lower()

    # Check for whole word matches
    return any(f' {bad_word.lower()} ' in f' {lower_text} ' for bad_word in bad_words)

def analyze_sentiment(text):
    # Check for bad words indicating potential anger
    if contains_bad_words(text):
        return 'angry'

    # Use sentiment analysis pipeline for other cases
    emotion_classifier = pipeline("sentiment-analysis")
    result = emotion_classifier(text)
    sentiment_label = result[0]['label']

    # Map sentiment labels to emotions
    if sentiment_label == 'POSITIVE':
        predicted_emotion = 'Happy'
    elif sentiment_label == 'NEGATIVE':
        predicted_emotion = 'Sad'
    else:
        predicted_emotion = 'Neutral'

    return predicted_emotion

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        audio_file = request.files['audio']
        text = recognize_text(audio_file)
        emotion = analyze_sentiment(text)
        return jsonify({'emotion': emotion})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
