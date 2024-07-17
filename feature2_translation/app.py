from flask import Flask, request, jsonify
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import os

app = Flask(__name__)

@app.route('/translate', methods=['POST'])
def translate():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please say something")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio"}), 400
    except sr.RequestError:
        return jsonify({"error": "Could not request results"}), 500

    translator = Translator()
    translated_text = translator.translate(text, src='en', dest='rw').text
    print(f"Translated text: {translated_text}")

    tts = gTTS(text=translated_text, lang='rw')
    tts.save("output.mp3")

    return jsonify({"translated_text": translated_text})

if __name__ == '__main__':
    app.run(debug=True)

