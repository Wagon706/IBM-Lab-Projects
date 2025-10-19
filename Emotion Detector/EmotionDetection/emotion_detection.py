import requests

def emotion_detector(text_to_analyze):
    if not text_to_analyze.strip():
        return {"anger": None, "disgust": None, "fear": None, "joy": None, "sadness": None, "dominant_emotion": None}

    url = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    headers = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    payload = {"raw_document": {"text": text_to_analyze}}
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 400:
        return {"anger": None, "disgust": None, "fear": None, "joy": None, "sadness": None, "dominant_emotion": None}

    emotions = response.json()["emotionPredictions"][0]["emotion"]
    dominant_emotion = max(emotions, key=emotions.get)

    return {"anger": emotions["anger"], "disgust": emotions["disgust"], "fear": emotions["fear"], "joy": emotions["joy"], "sadness": emotions["sadness"], "dominant_emotion": dominant_emotion}
