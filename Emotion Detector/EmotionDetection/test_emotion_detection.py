from emotion_detection import emotion_detector

def get_dominant_emotion(result):
    emotions = result["emotionPredictions"][0]["emotion"]
    return max(emotions, key=emotions.get)


tests = [
    ("I am glad this happened", "joy"),
    ("I am really mad about this", "anger"),
    ("I feel disgusted just hearing about this", "disgust"),
    ("I am so sad about this", "sadness"),
    ("I am really afraid that this will happen", "fear")
]

for text, expected in tests:
    result = emotion_detector(text)
    dominant = get_dominant_emotion(result)
    print(f"Statement: '{text}' Dominant Emotion: '{dominant}'")
    assert dominant == expected, f"Test failed for '{text}'"

print("All tests passed!")

