from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
stemmer = SnowballStemmer('english')


def stem_phrase(phrase):
    words = phrase.split(" ")
    stemmed_words = []
    for word in words:
        stemmed_word = stemmer.stem(word)
        stemmed_words.append(stemmed_word)
    stemmed_phrase = " ".join(stemmed_words)
    return stemmed_phrase


# you can add if you find some additional fix or cleaning
def clean_text(phrase):
    phrase = phrase.replace("n't", "not")
    phrase = phrase.replace("it's", "it is")
    return phrase


def cleaning(data):
    tqdm.pandas(desc="Stemming...(train)")
    data['text'] = data['text'].progress_apply(stem_phrase)
    return data
