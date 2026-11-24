import fasttext

nsfw_model = fasttext.load_model("local-shared-data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")
toxic_speech_model = fasttext.load_model("local-shared-data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")

def classify_nsfw(text: str) -> tuple[str, float]:
    text = text.replace("\n", " ")
    predictions = nsfw_model.predict(text)
    return (predictions[0][0].replace("__label__", ""), predictions[1][0])

def classify_toxic_speech(text: str) -> tuple[str, float]:
    text = text.replace("\n", " ")
    predictions = toxic_speech_model.predict(text)
    return (predictions[0][0].replace("__label__", ""), predictions[1][0])
