import nltk
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
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
    phrase = phrase.replace("n't", " not")
    phrase = phrase.replace("it's", "it is")
    phrase = phrase.replace("'v", " have")
    return phrase


def cleaning(data):
    data.dropna()
    tqdm.pandas(desc="Stemming...")
    data['text'] = data['text'].progress_apply(stem_phrase)
    data['text'] = data['text'].progress_apply(clean_text)
    return data


# easy one hot encoding
def phrase_one_hot_encode(data, num=30000):
    vectorizer = CountVectorizer(max_features=num, min_df=1, ngram_range=(1, 2))
    vectorizer.fit(data['text'])
    return vectorizer.transform(data['text'])


def phrase_tf_idf_encode(data, num=30000):
    vectorizer = CountVectorizer(max_features=num, min_df=1, ngram_range=(1, 2))
    transformer = TfidfTransformer(smooth_idf=False)
    vectorizer.fit(data['text'])
    counts = vectorizer.transform(data['text'])
    tfidf = transformer.fit_transform(counts)
    return tfidf



