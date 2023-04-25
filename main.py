# main for doing classification of performance by individuals and teams based on text features,
# message counts and frequencies, part of speech taggins, and sentiment features,

import nltk
import GetFeatures
import RunModels
import CategorizeData
from CleanDataset import clean_dataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# downloads necessary corpora
#nltk.download()

# creates clean data pickle
#clean_dataset()

# adds performance categories to data and derives some features
#CategorizeData.categorize_data("clean_data.pickle", "categorized_data.pickle", True)
#CategorizeData.categorize_data_person("clean_data.pickle", "categorized_individual_data.pickle", True)

def classify(pickle_file, is_individual, use_sentiment):
    X, y = GetFeatures.tf_idf(pickle_file, is_individual)
    y_cat = []
    for i in y:
        if i == 1:
            y_cat.append("Low Performance")
        elif i ==2:
            y_cat.append("Mid Performance")
        elif i == 3:
            y_cat.append("High Performance")
    y_cat_u = np.unique(y_cat)
    colors = ["#1B9E77", "#D95F02", "#7570B3"]

    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(X.toarray(), y)
    X_lda = np.array(X_lda)

    x_plot = X_lda[:, 0]
    y_plot = X_lda[:, 1]
    fig, ax = plt.subplots(figsize=(8, 8))
    for cat, color in zip(y_cat_u, colors):
        idxs = np.where(np.array(y_cat) == cat)
        print(idxs)
        ax.scatter(x_plot[idxs], y_plot[idxs], c=color, label=cat, s=50)
    ax.legend()
    plt.title("LDA components of tf_idf")
    plt.show()

    X_turn_taking = GetFeatures.turn_taking(pickle_file)
    X_turn_taking = np.array(X_turn_taking)
    sc = StandardScaler()
    X_turn_taking = sc.fit_transform(X_turn_taking)
    lda2 = LDA(n_components=2)
    X_turn_taking_lda = lda2.fit_transform(X_turn_taking, y)
    x_plot = X_turn_taking_lda[:, 0]
    y_plot = X_turn_taking_lda[:, 1]
    fig, ax = plt.subplots(figsize=(8, 8))
    for cat, color in zip(y_cat_u, colors):
        idxs = np.where(np.array(y_cat) == cat)
        print(idxs)
        ax.scatter(x_plot[idxs], y_plot[idxs], c=color, label=cat, s=50)
    ax.legend()
    plt.title("LDA components of turn taking")
    plt.show()

    X_pos = GetFeatures.part_of_speech(pickle_file)
    X_pos = np.array(X_pos)
    sc2 = StandardScaler()
    X_pos = sc2.fit_transform(X_pos)
    lda3 = LDA(n_components=2)
    X_pos_lda = lda3.fit_transform(X_pos, y)
    x_plot = X_pos_lda[:, 0]
    y_plot = X_pos_lda[:, 1]
    fig, ax = plt.subplots(figsize=(8, 8))
    for cat, color in zip(y_cat_u, colors):
        idxs = np.where(np.array(y_cat) == cat)
        print(idxs)
        ax.scatter(x_plot[idxs], y_plot[idxs], c=color, label=cat, s=50)
    ax.legend()
    plt.title("LDA components of part of speech tagging")
    plt.show()

    if(use_sentiment):
        X_sentiment = GetFeatures.sentiment(pickle_file)
        X_sentiment = np.array(X_sentiment)
        lda4 = LDA(n_components=2)
        X_sentiment_lda = lda4.fit_transform(X_sentiment, y)
        x_plot = X_sentiment_lda[:, 0]
        y_plot = X_sentiment_lda[:, 1]
        fig, ax = plt.subplots(figsize=(8, 8))
        for cat, color in zip(y_cat_u, colors):
            idxs = np.where(np.array(y_cat) == cat)
            print(idxs)
            ax.scatter(x_plot[idxs], y_plot[idxs], c=color, label=cat, s=50)
        ax.legend()
        plt.title("LDA components of sentiment scores")
        plt.show()

        X_total = np.concatenate((X_lda, X_turn_taking_lda, X_pos_lda, X_sentiment_lda), axis=1)
        print("classification for tf_idf, message frequency, part of speech, and sentiment features\n")
    else:
        X_total = np.concatenate((X_lda, X_turn_taking_lda, X_pos_lda), axis=1)
        print("classification for tf_idf, message frequency, and part of speech features\n")

    random_seed = 4222023

    data = [[],
            [],
            [],
            [],
            [],
            [],
            []]
    maxes = []
    train_percentages = [0.3, 0.4, 0.5, 0.6, 0.7]
    for train_percent in train_percentages:
        score, conf = RunModels.run_model("svm_rbf", X_total, y, train_percent, random_seed)
        print("Score for svm_rbf with train size ", train_percent, ": ", score)
        print("Confusion Matrix for svm_rbf: ")
        print(conf)
        score = score * 100
        max = score
        data[0].append("%.2f" % score)

        score, conf = RunModels.run_model("svm_poly", X_total, y, train_percent, random_seed)
        print("Score for svm_poly with train size ", train_percent, ": ", score)
        print("Confusion Matrix for svm_poly: ")
        print(conf)
        score = score * 100
        if score > max:
            max = score
        data[1].append("%.2f" % score)

        score, conf = RunModels.run_model("svm_linear", X_total, y, train_percent, random_seed)
        print("Score for svm_linear with train size ", train_percent, ": ", score)
        print("Confusion Matrix for svm_linear: ")
        print(conf)
        score = score * 100
        if score > max:
            max = score
        data[2].append("%.2f" % score)

        score, conf = RunModels.run_model("guass_nb", X_total, y, train_percent, random_seed)
        print("Score for guass_nb with train size ", train_percent, ": ", score)
        print("Confusion Matrix for guass_nb: ")
        print(conf)
        score = score * 100
        if score > max:
            max = score
        data[3].append("%.2f" % score)

        score, conf = RunModels.run_model("mlp", X_total, y, train_percent, random_seed)
        print("Score for mlp with train size ", train_percent, ": ", score)
        print("Confusion Matrix for mlp: ")
        print(conf)
        score = score * 100
        if score > max:
            max = score
        data[4].append("%.2f" % score)

        score, conf = RunModels.run_model("rf", X_total, y, train_percent, random_seed)
        print("Score for rf with train size ", train_percent, ":", score)
        print("Confusion Matrix for rf: ")
        print(conf)
        score = score * 100
        if score > max:
            max = score
        data[6].append("%.2f" % score)

        score, conf = RunModels.run_model("dt", X_total, y, train_percent, random_seed)
        print("Score for dt with train size ", train_percent, ":", score)
        print("Confusion Matrix for dt: ")
        print(conf)
        score = score * 100
        if score > max:
            max = score
        data[5].append("%.2f" % score)

        maxes.append(max)

    columns = ('30%', '40%', '50%',
               '60%', '70%')
    rows = ['SVM (RBF)', 'SVM (Poly)', 'SVM (Linear)', 'Naive Bayes (Guassian)', 'MLP',
            'Decision Tree', 'Random Forest']

    # Get some pastel shades for the colors
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for
    # the line plots.
    y_offset = np.zeros(len(columns))

    # Plot line plots and create text labels
    # for the table
    cell_text = []
    for row in range(n_rows):
        floats = [float(x) for x in data[row]]
        plt.plot(index, floats, color=colors[row])
        y_offset = data[row]
        cell_text.append([x for x in y_offset])

    # Reverse colors and text labels to display
    # the last value at the top.

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          rowColours=colors,
                          colLabels=columns,
                          loc='bottom')
    #the_table.auto_set_font_size(False)
    the_table.set_fontsize(16)

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.27)

    plt.ylabel("Classification Accuracy")
    plt.xticks([])
    plt.title('Classification Accuracy of Different Models at Different Training Percentages')

    plt.show()

    print("Done")

    return maxes

# run model training and classifications on both categorized data sets
maxes_sentiment_team = classify("categorized_data.pickle", False, True)
maxes_sentiment_individual = classify("categorized_individual_data.pickle", True, True)

print(maxes_sentiment_team)
print(maxes_sentiment_individual)
