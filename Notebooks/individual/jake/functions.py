# functions file for reusing functions accross notebooks

import pandas as pd
import numpy as np

import re
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
