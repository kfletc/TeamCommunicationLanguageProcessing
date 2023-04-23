# file for doing tf_idf or bag of words for categorized data and also for extracting features from categorized data

import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# return sparse matrix bag of words for text data
def bag_of_words(pickle_file):
    vectorizer = CountVectorizer()

    with open(pickle_file, 'rb') as categorized_data:
        data = pickle.load(categorized_data)

    data_df = pd.DataFrame(data)
    docs = np.array(data_df["words"])
    bag = vectorizer.fit_transform(docs)
    y = np.array(data_df["category"])

    return bag, y

# return sparse matrix of tf_idf for text data
# also produces bar graphs of most important features for the low and high performance classes
def tf_idf(pickle_file, is_individual):
    tfidf = TfidfVectorizer()

    with open(pickle_file, 'rb') as categorized_data:
        data = pickle.load(categorized_data)

    data_df = pd.DataFrame(data)
    docs = np.array(data_df["words"])
    X_tfidf = tfidf.fit_transform(docs)
    y = np.array(data_df["category"])

    X_array = X_tfidf.toarray()
    if is_individual == False:
        tf_idf_totals_low = [x for x in X_array[0]]
        for i in range(1, 14):
            tf_idf_totals_low = [x1 + x2 for x1, x2 in zip(tf_idf_totals_low, X_array[i])]
        top_feat_df = top_tfidf_feats(tf_idf_totals_low, tfidf.get_feature_names_out())
        ax = top_feat_df.plot.bar(x='feature', y='tfidf', rot=0)
        plt.ylabel("Sum of TF_IDF scores")
        plt.title("Most important TF_IDF features for low performance teams")

        tf_idf_totals_low = [x for x in X_array[28]]
        for i in range(29, 41):
            tf_idf_totals_low = [x1 + x2 for x1, x2 in zip(tf_idf_totals_low, X_array[i])]
        top_feat_df = top_tfidf_feats(tf_idf_totals_low, tfidf.get_feature_names_out())
        ax = top_feat_df.plot.bar(x='feature', y='tfidf', rot=0)
        plt.ylabel("Sum of TF_IDF scores")
        plt.title("Most important TF_IDF features for high performance teams")

    else:
        tf_idf_totals_low = [x for x in X_array[0]]
        for i in range(1, 56):
            tf_idf_totals_low = [x1 + x2 for x1, x2 in zip(tf_idf_totals_low, X_array[i])]
        top_feat_df = top_tfidf_feats(tf_idf_totals_low, tfidf.get_feature_names_out())
        fig, ax = plt.subplots()
        ax = top_feat_df.plot.bar(x='feature', y='tfidf', rot=0)
        plt.ylabel("Sum of TF_IDF scores")
        plt.title("Most important TF_IDF features for low performance individuals")

        tf_idf_totals_low = [x for x in X_array[112]]
        for i in range(113, 164):
            tf_idf_totals_low = [x1 + x2 for x1, x2 in zip(tf_idf_totals_low, X_array[i])]
        top_feat_df = top_tfidf_feats(tf_idf_totals_low, tfidf.get_feature_names_out())
        fig, ax = plt.subplots()
        ax = top_feat_df.plot.bar(x='feature', y='tfidf', rot=0)
        plt.ylabel("Sum of TF_IDF scores")
        plt.title("Most important TF_IDF features for high performance individuals")

    return X_tfidf, y

# returns top features for a row of tf_idf
# used with summed rows
def top_tfidf_feats(row, features, top_n=20):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

# return matrix of turn_taking features
def turn_taking(pickle_file):
    with open(pickle_file, 'rb') as categorized_data:
        data = pickle.load(categorized_data)

    X_turn_taking = data[
        ["Total Turns", "Message Frequency Sim 0", "Message Frequency Sim 1", "Message Frequency Sim 2",
         "Message Frequency Sim 3"]].to_numpy()
    return X_turn_taking

# return matrix of part of speech features
def part_of_speech(pickle_file):
    with open(pickle_file, 'rb') as categorized_data:
        data = pickle.load(categorized_data)

    X_pos = data[
        ["NN", "JJ", "VB", "RB", "VBP", "IN", "VBN", "CD", "JJS", "VBD", "NNS", "FW", "VBG", "MD", "RP", "VBZ", "JJR",
         "RBR", "TO", "CC", "DT", "WP", "WDT", "PRP", "UH", "WRB", "NNP"]].to_numpy()
    return X_pos

# return matrix of sentiment features
def sentiment(pickle_file):
    with open(pickle_file, 'rb') as categorized_data:
        data = pickle.load(categorized_data)

    X_sentiment = data[
        ["Sentiment Negative", "Sentiment Neutral", "Sentiment Positive", "Sentiment Compound"]].to_numpy()
    return X_sentiment
