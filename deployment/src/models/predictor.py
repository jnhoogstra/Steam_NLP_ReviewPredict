import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import string
import nltk
nltk.download("stopwords")
nltk.download("wordnet")

from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def CleanText(reviews):
    stopwords = nltk.corpus.stopwords.words('english')
    punct = string.punctuation
    lemma = nltk.WordNetLemmatizer()
    
    reviews = "".join([word for word in reviews if word not in string.punctuation])
    tokens = re.split('\W+', reviews)
    reviews = [lemma.lemmatize(word) for word in tokens if word not in stopwords]
    
    return reviews

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]

def create_usable(x):
    return ' '.join(x["app_tags"]) + ' ' + ' '

def get_recommendations(title, score, df):
    thanos = df
    features = ["app_tags"]
    for feature in features:
        thanos[feature] = thanos[feature].apply(literal_eval)
        thanos[feature] = thanos[feature].apply(clean_data)
    
    thanos["usable"] = thanos.apply(create_usable, axis=1)
    title_and_tags = thanos.drop(['steamid', 'appid', 'app_tags', 'review', 'fps', 'voted_up'], axis=1)
    title_and_tags = title_and_tags.drop_duplicates()
    title_and_tags["app_title"].unique()
    
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(title_and_tags['usable'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    recommend = title_and_tags.reset_index()
    indices = pd.Series(recommend.index, index=recommend['app_title'])
    
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    if score == 0:
        sim_scores = sim_scores[156:167]
    elif score == 1:
        sim_scores = sim_scores[1:11]

    # Get the movie indices
    game_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return recommend['app_title'].iloc[game_indices]


def get_prediction(feature_values):
    """ Given a list of feature values, return a prediction made by the model"""
    
    loaded_model = un_pickle_model()
    
    # Model is expecting vectorized text, so we first need to transform the review
    df = pd.read_csv("src/models/thanos.csv")
    vector = TfidfVectorizer(analyzer=CleanText, ngram_range=(2, 2))
    vector.fit(df["review"])
    
    data = vector.transform(feature_values["review"])
    
    data = pd.DataFrame(data.toarray())
    data.columns = vector.get_feature_names()
    
    # Now the model can make a prediction from this vectorized text
    predictions = loaded_model.predict(data)
    
    
    # here we get what we need from the recommendation system
    # converting predictions to integers
    if predictions == True:
        score = 1
    else:
        score = 0
    title = feature_values["game_title"]
        
    recommendations = get_recommendations(title, score, df).to_list()
    
    # We are only making a single prediction, so return the 0-th value
    #predictions[0], 
    return recommendations


def un_pickle_model():
    """ Load the model from the .pkl file """
    with open("src/models/nlp_model.pkl", "rb") as model_file:
        loaded_model = pickle.load(model_file)
        
    return loaded_model