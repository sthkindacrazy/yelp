import nltk
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import re

nltk.download()
stop_words = set(stopwords.words('english'))
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
    data = data.dropna()
    tqdm.pandas(desc="Stemming...")
    data['text'] = data['text'].progress_apply(stem_phrase)
    data['text'] = data['text'].progress_apply(clean_text)
    return data


# easy one hot encoding
def phrase_one_hot_encode(train_data, test_data, num=30000):
    vectorizer = CountVectorizer(max_features=num, min_df=1, ngram_range=(1, 2), binary=True)
    vectorizer.fit(train_data['text'])
    return vectorizer.transform(train_data['text']), vectorizer.transform(test_data['text'])


def phrase_tf_idf_encode(data, num=30000):
    vectorizer = CountVectorizer(max_features=num, min_df=1, ngram_range=(1, 2))
    transformer = TfidfTransformer(smooth_idf=False)
    vectorizer.fit(data['text'])
    counts = vectorizer.transform(data['text'])
    tfidf = transformer.fit_transform(counts)
    return tfidf

#############
##############revised part 
def cln(data):
    cln_text =[]
    for sentence in data["text"]:
        cln_text.append(re.sub("[!&\'()*+,-./:;<=>?@[\\]^_`{|}~]", "", sentence))
    data["no_punc_text"] = cln_text
    return data

def rm_stopwords(data):
    data_cl = cln(data)
    filtered_sentence=[]
    for sentence in data_cl["no_punc_text"]:
        word_tokens = word_tokenize(sentence) 
        sentence_tem = [w for w in word_tokens if not w in stop_words]
        filtered_sentence.append(sentence_tem)
    data["cln_text"] = filtered_sentence
    clc_sentence = []
    for sentence in data["cln_text"]:
        clc_sentence.append(" ".join(sentence))
    data["cln_text"]=clc_sentence
    return data



