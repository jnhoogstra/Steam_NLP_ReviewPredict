# functions file for reusing functions accross notebooks

import pandas as pd
import numpy as np
import time

import re
import requests
from bs4 import BeautifulSoup

import string
import nltk
nltk.download("stopwords")
nltk.download("wordnet")

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score


# make sure your model is fit before scoring
def ScoreModel(model, X, y):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    recall = recall_score(y, preds)
    precision = precision_score(y, preds)
    rockout = roc_auc_score(y, preds)
    
    print("Accuracy:  ", acc)
    print("F1 Score:  ", f1)
    print("Recall:    ", recall)
    print("Precision: ", precision)
    print("ROC_AUC:   ", rockout)

    
    
    
def CleanText(reviews):
    stopwords = nltk.corpus.stopwords.words('english')
    punct = string.punctuation
    lemma = nltk.WordNetLemmatizer()
    
    reviews = "".join([word for word in reviews if word not in string.punctuation])
    tokens = re.split('\W+', reviews)
    reviews = [lemma.lemmatize(word) for word in tokens if word not in stopwords]
    
# use function like this in your notebook
# df['cleaned_text'] = df['text'].apply(lambda x: CleanText(x.lower()))
    
    return reviews





###########################################################
#### webscrape stuff ####
#### fetching titles and tags from steam using appid ####

## internal functions, not meant to be exported ##
def get_title(soup):
    title = soup.find('div', class_="apphub_AppName")
    return title
    
def get_tags(soup):
    warning = soup.find('div', class_="glance_tags popular_tags")
    tags = [p.text for p in warning.findAll('a', class_="app_tag")]
    
    for index in range(len(tags)):
        tags[index] = tags[index].replace("\t", "")
        tags[index] = tags[index].replace("\r\n", "")
    return tags



#### MAIN FUNCTION ####
def FetchTitlesTags(df):
    game_ids = list(df["appid"].unique())
    url_list = []
    
    for each in game_ids:
        url = "https://store.steampowered.com/app/{}/".format(each)
        url_list.append(url)
        
    titles = []
    tags = []

    for url in url_list:
        html_page = requests.get(url)
        soup = BeautifulSoup(html_page.content, 'html.parser')
        titles += get_title(soup)
        tags.append(get_tags(soup))
        time.sleep(1.5)
        
    return titles, tags