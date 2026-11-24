import nltk


def gopher_quality_filter(text: str) -> bool:
    """检查词数、平均词长、行尾省略号和含字母词的比例。"""
    words = nltk.word_tokenize(text)

    if len(words) < 50 or len(words) > 100000:
        return False
    
    mean_word_length = sum(len(word) for word in words) / len(words)
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    lines = text.splitlines()
    ellipsis_lines = sum(1 for line in lines if line.strip().endswith("..."))
    if len(lines) > 0 and (ellipsis_lines / len(lines)) > 0.3:
        return False
    
    alphabetic_words = sum(1 for word in words if any(char.isalpha() for char in word))
    if len(words) > 0 and (alphabetic_words / len(words)) < 0.8:
        return False
    
    return True

if __name__ == "__main__":
    sample_text = "Hello, world! 123"
    words = nltk.word_tokenize(sample_text)
    print(words)