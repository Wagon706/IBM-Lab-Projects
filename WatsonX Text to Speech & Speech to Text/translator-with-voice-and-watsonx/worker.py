# worker.py

import requests
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# setup for the WatsonX model
PROJECT_ID = "skills-network"
MODEL_ID = "mistralai/mistral-medium-2505"

credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    # "apikey": API_KEY  # add your key here if you have one
}

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1024
}

model = Model(
    model_id=MODEL_ID,
    params=parameters,
    credentials=credentials,
    project_id=PROJECT_ID
)

# turns speech into text
def speech_to_text(audio_binary):
    if not audio_binary:
        return ""
    try:
        base_url = 'https://sn-watson-stt.labs.skills.network'
        api_url = base_url + '/speech-to-text/api/v1/recognize'
        params = {'model': 'en-US_Multimedia'}

        response = requests.post(api_url, params=params, data=audio_binary).json()
        results = response.get('results', [])
        if results:
            text = results[-1]['alternatives'][0]['transcript']
            print("Recognized text:", text)
            return text
        else:
            print("No results from STT API")
            return ""
    except Exception as e:
        print("Error in speech_to_text:", e)
        return ""

# turns text into speech (placeholder for now)
def text_to_speech(text, voice="en-GB_KateV3Voice"):
    if not text:
        return b""
    try:
        # just encode it for now so it doesn’t crash
        audio_bytes = text.encode('utf-8')
        return audio_bytes
    except Exception as e:
        print("Error in text_to_speech:", e)
        return b""

# processes the message using WatsonX model
def watsonx_process_message(user_message):
    if not user_message:
        return ""

    try:
        response = model.generate([user_message])
        print("WatsonX raw response:", response)  # helps see what the model returns

        text = ""

        # check if it’s a list of dicts with generated_text
        if isinstance(response, list) and len(response) > 0:
            # sometimes the model output may be inside a key
            if 'generated_text' in response[0]:
                text = response[0]['generated_text']
            elif 'results' in response[0]:
                # for some models that wrap output in 'results'
                text = response[0]['results'][0].get('generated_text', "")
        
        # fallback message if text is empty
        if not text.strip():
            text = "Sorry, I didn’t get that."

        print("WatsonX final text:", text)
        return text

    except Exception as e:
        print("Error calling WatsonX model:", e)
        return "Something went wrong while getting a response."
