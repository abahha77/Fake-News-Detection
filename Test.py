import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

# Load vectorizer used in training
vectorizer=pickle.load(open('tfidf_vect.pkl', 'rb'))



# Preprocessing functions
contractions_dict = {"ain't": "are not", "'s": " is", "aren't": "are not"}
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def expand_contractions(text, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

stop_words = set(stopwords.words('english'))
stop_words.add('subject')
stop_words.add('http')

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])

stemmer = PorterStemmer()

def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


# Preprocess new article
def preprocess(article):
    article = expand_contractions(article)
    article = article.lower()
    article = re.sub('[%s]' % re.escape(string.punctuation), '', article)
    article = re.sub('W*dw*', '', article)
    article = re.sub('(http[s]?S+)|(w+.[A-Za-z]{2,4}S*)', 'urladd', article)
    article = remove_stopwords(article)
    article = stem_words(article)
    article = lemmatize_words(article)
    return article

    article_vector = vectorizer.transform([article])
    return article_vector


# Test predict function
def predict(article):
    article = preprocess(article)
    article_vector = vectorizer.transform([article])
    prediction = model.predict(article_vector)[0]
    if prediction == 1:
        return "Real News"
    else:
        return "Fake News"

article="C:/Users\Ahmed Baha\Desktop\NLProject\news.csv""

predict(article)