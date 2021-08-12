import pandas as pd
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import string
import nltk
nltk.download("stopwords")
nltk.download("wordnet")


def CleanText(reviews):
    stopwords = nltk.corpus.stopwords.words('english')
    punct = string.punctuation
    lemma = nltk.WordNetLemmatizer()
    
    reviews = "".join([word for word in reviews if word not in string.punctuation])
    tokens = re.split('\W+', reviews)
    reviews = [lemma.lemmatize(word) for word in tokens if word not in stopwords]
    
    return reviews


def get_prediction(feature_values):
    """ Given a list of feature values, return a prediction made by the model"""
    
    loaded_model = un_pickle_model()
    
    # Model is expecting vectorized text, so we first need to transform the review
    df = pd.read_csv("src/models/thanos.csv")
    vector = TfidfVectorizer(analyzer=CleanText, ngram_range=(2, 2))
    vector.fit(df["review"])
    
    data = vector.transform(feature_values["review"])
    
    #vectorized = loaded_vector.transform(feature_values["review"])
    data = pd.DataFrame(data.toarray())
    data.columns = vector.get_feature_names()
    
    # Now the model can make a prediction from this vectorized text
    predictions = loaded_model.predict(data)
    
    
    # here we get what we need from the recommendation system
    
    
    
    # We are only making a single prediction, so return the 0-th value
    return predictions[0]


def un_pickle_model():
    """ Load the model from the .pkl file """
    with open("src/models/nlp_model.pkl", "rb") as model_file:
        loaded_model = pickle.load(model_file)
        
    #with open("src/models/vector.pkl", "rb") as vector_file:
    #    loaded_vector = pickle.load(vector_file)
        
    #with open("src/models/clean_text.pkl", "rb") as clean_text_file:
     #   CleanText = pickle.load(clean_text_file)
        
    return loaded_model#, loaded_vector, CleanText