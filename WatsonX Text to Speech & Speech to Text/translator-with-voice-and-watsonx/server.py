import base64
import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from worker import speech_to_text, text_to_speech, watsonx_process_message

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/speech-to-text', methods=['POST'])
def speech_to_text_route():
    print("Processing Speech-to-Text...")

    audio_binary = request.data
    if not audio_binary:
        return app.response_class(
            response=json.dumps({'error': 'No audio data received.'}),
            status=400,
            mimetype='application/json'
        )

    text = speech_to_text(audio_binary)
    print("STT result:", text)

    response = app.response_class(
        response=json.dumps({'text': text}),
        status=200,
        mimetype='application/json'
    )
    return response


@app.route('/process-message', methods=['POST'])
def process_message_route():
    data = request.get_json(force=True)
    user_message = data.get('userMessage', '')
    voice = data.get('voice', 'en-GB_KateV3Voice')

    print('User message:', user_message)
    print('Voice:', voice)

    # make sure the message isn’t empty
    if not user_message.strip():
        return app.response_class(
            response=json.dumps({'error': 'Empty message received.'}),
            status=400,
            mimetype='application/json'
        )

    # get a text response from WatsonX
    watsonx_response_text = watsonx_process_message(user_message) or ""

    # remove any empty lines
    watsonx_response_text = os.linesep.join(
        [s for s in watsonx_response_text.splitlines() if s]
    )

    # turn the response text into speech
    watsonx_response_speech = text_to_speech(watsonx_response_text, voice) or b""
    watsonx_response_speech = base64.b64encode(watsonx_response_speech).decode('utf-8')

    # send both the text and speech back to the frontend
    response_data = {
        "watsonxResponseText": watsonx_response_text,
        "watsonxResponseSpeech": watsonx_response_speech
    }

    print("WatsonX text:", watsonx_response_text[:100], "..." if len(watsonx_response_text) > 100 else "")
    print("Response ready.")

    response = app.response_class(
        response=json.dumps(response_data),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == "__main__":
    app.run(port=8000, host='0.0.0.0')
