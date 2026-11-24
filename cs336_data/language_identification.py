import fasttext


model = fasttext.load_model("local-shared-data/classifiers/lid.176.bin")

def identify_language(text: str) -> tuple[str, float]:
    text = text.replace("\n", " ")
    predictions = model.predict(text)
    return (predictions[0][0].replace("__label__", ""), predictions[1][0])


if __name__ == "__main__":
    sample_text = "This is a sample text to identify the language. \n It should be in English."
    language, confidence = identify_language(sample_text)
    print(f"Identified Language: {language}, Confidence: {confidence}")