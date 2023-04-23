# for creating cvs and pickle files with teams and individuals classified into performance categories
# based on mode and average of SA scores. Also preprocesses text and other derived features such as
# message frequency, part of speech tagging, and sentiment analysis. Groups all text data with all of these
# features
from nltk import word_tokenize
import pickle
import math
import nltk
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
import statistics
import pandas as pd
import datetime
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import matplotlib.pyplot as plt

lemmatizer = WordNetLemmatizer()

# calculates sum of mode and averages of SA scores for a team
def sa_average(sim):
    i = 0
    j = 0
    total = 0
    scores = []
    for sa_score in sim['SA']:
        j += 1
        if not math.isnan(sa_score):
            total += int(sa_score)
            scores.append(int(sa_score))
            i += 1
    median = statistics.median(scores)
    mode = statistics.mode(scores)
    average_no0 = float(total / i)
    true_average = float(total / j)
    return mode + average_no0 + true_average

# calculates sum of mode and averages of SA scores of an individual
def sa_average_person(sim, person):
    i = 0
    j = 0
    total = 0
    scores = []
    for sa_score, person_num in zip(sim['SA'], sim["Turn Taking"]):
        if int(person_num) == person:
            j += 1
            if not math.isnan(sa_score):
                total += int(sa_score)
                scores.append(int(sa_score))
                i += 1
    if len(scores) > 0:
        median = statistics.median(scores)
        mode = statistics.mode(scores)
        average_no0 = float(total / i)
    else:
        median = 0
        mode = 0
        average_no0 = 0
    true_average = float(total / j)
    return mode + average_no0 + true_average

# calculates frequency of messages for a team in each simulation
def message_frequency(sim):
    min_time0 = 10000.0
    max_time0 = 0.0
    min_time1 = 10000.0
    max_time1 = 0.0
    min_time2 = 10000.0
    max_time2 = 0.0
    min_time3 = 10000.0
    max_time3 = 0.0
    num0 = 0
    num1 = 0
    num2 = 0
    num3 = 0
    for sim_num, time_formatted in zip(sim["Simulation"], sim["Time"]):
        if isinstance(time_formatted, datetime.time):
            seconds = (time_formatted.hour * 60.0 + time_formatted.minute) * 60.0 + time_formatted.second
            if sim_num == 0:
                num0 += 1
                if seconds < min_time0:
                    min_time0 = seconds
                if seconds > max_time0:
                    max_time0 = seconds
            elif sim_num == 1:
                num1 += 1
                if seconds < min_time1:
                    min_time1 = seconds
                if seconds > max_time1:
                    max_time1 = seconds
            elif sim_num == 2:
                num2 += 1
                if seconds < min_time2:
                    min_time2 = seconds
                if seconds > max_time2:
                    max_time2 = seconds
            elif sim_num == 3:
                num3 += 1
                if seconds < min_time3:
                    min_time3 = seconds
                if seconds > max_time3:
                    max_time3 = seconds
    if (max_time0 - min_time0) == 0:
        freq0 = 0
    else:
        freq0 = num0 / (max_time0 - min_time0)
    if (max_time1 - min_time1) == 0:
        freq1 = 0
    else:
        freq1 = num1 / (max_time1 - min_time1)
    if (max_time2 - min_time2) == 0:
        freq2 = 0
    else:
        freq2 = num2 / (max_time2 - min_time2)
    if (max_time3 - min_time3) == 0:
        freq3 = 0
    else:
        freq3 = num3 / (max_time3 - min_time3)
    return freq0, freq1, freq2, freq3

# calculates message frequencies of an individual in each simulation
def message_frequency_person(sim, person):
    min_time0 = 10000.0
    max_time0 = 0.0
    min_time1 = 10000.0
    max_time1 = 0.0
    min_time2 = 10000.0
    max_time2 = 0.0
    min_time3 = 10000.0
    max_time3 = 0.0
    num0 = 0
    num1 = 0
    num2 = 0
    num3 = 0
    for sim_num, time_formatted, person_num in zip(sim["Simulation"], sim["Time"], sim["Turn Taking"]):
        if person == str(person_num):
            if isinstance(time_formatted, datetime.time):
                seconds = (time_formatted.hour * 60.0 + time_formatted.minute) * 60.0 + time_formatted.second
                if sim_num == 0:
                    num0 += 1
                    if seconds < min_time0:
                        min_time0 = seconds
                    if seconds > max_time0:
                        max_time0 = seconds
                elif sim_num == 1:
                    num1 += 1
                    if seconds < min_time1:
                        min_time1 = seconds
                    if seconds > max_time1:
                        max_time1 = seconds
                elif sim_num == 2:
                    num2 += 1
                    if seconds < min_time2:
                        min_time2 = seconds
                    if seconds > max_time2:
                        max_time2 = seconds
                elif sim_num == 3:
                    num3 += 1
                    if seconds < min_time3:
                        min_time3 = seconds
                    if seconds > max_time3:
                        max_time3 = seconds
    if (max_time0 - min_time0) == 0:
        freq0 = 0
    else:
        freq0 = num0 / (max_time0 - min_time0)
    if (max_time1 - min_time1) == 0:
        freq1 = 0
    else:
        freq1 = num0 / (max_time1 - min_time1)
    if (max_time2 - min_time2) == 0:
        freq2 = 0
    else:
        freq2 = num0 / (max_time2 - min_time2)
    if (max_time3 - min_time3) == 0:
        freq3 = 0
    else:
        freq3 = num0 / (max_time3 - min_time3)
    return freq0, freq1, freq2, freq3

# for sorting
def value_getter(item):
    return item[1]

# displays graphs of stats and creates pickled dataframe of all teams and their features and performance category
def categorize_data(pickle_file1, pickle_file2, use_sentiment):
    with open(pickle_file1, 'rb') as dataset_pickle:
        data = pickle.load(dataset_pickle)

    sim_names = list(data.keys())

    sorted_scores = {}
    for name in sim_names:
        sim = data[name]
        sorted_scores[name] = sa_average(sim)

    sorted_scores = dict(sorted(sorted_scores.items(), key=value_getter))

    count = 0
    new_data = {
        "words": [],
        "category": [],
        "SA Average": [],
        "Total Turns": [],
        "Message Frequency Sim 0": [],
        "Message Frequency Sim 1": [],
        "Message Frequency Sim 2": [],
        "Message Frequency Sim 3": []
    }

    if(use_sentiment):
        new_data["Sentiment Negative"] = []
        new_data["Sentiment Neutral"] = []
        new_data["Sentiment Positive"] = []
        new_data["Sentiment Compound"] = []

    low_score0 = [0, 0, 0, 0]
    low_score1 = [0, 0, 0, 0]
    low_score2 = [0, 0, 0, 0]
    low_score3 = [0, 0, 0, 0]

    high_score0 = [0, 0, 0, 0]
    high_score1 = [0, 0, 0, 0]
    high_score2 = [0, 0, 0, 0]
    high_score3 = [0, 0, 0, 0]

    low_turn_taking = 0
    mid_turn_taking = 0
    high_turn_taking = 0

    low_sentiment = 0
    mid_sentiment = 0
    high_sentiment = 0

    wordset = []
    for name in sorted_scores.keys():
        count += 1
        sim = data[name]
        total_words = []
        msg_count = 0
        neg_sent = 0
        neu_sent = 0
        pos_sent = 0
        com_sent = 0
        for msg in sim["Total"]:
            msg_count += 1
            msg = str(msg)
            msg = msg.lower()
            msg_words = word_tokenize(msg)
            msg_words_lemmatized = [lemmatizer.lemmatize(word) for word in msg_words]
            stop_words = stopwords.words('english')
            all_words = words.words()
            stop_words += [".", "?", ",", "!", "\'", "..."]
            msg_words_nostop = [word for word in msg_words_lemmatized if word not in stop_words]
            msg_words_cleaned = [word for word in msg_words_nostop if word in all_words]
            total_words += msg_words_cleaned

            # add sentiment
            if use_sentiment:
                sianalyzer = SentimentIntensityAnalyzer()
                sentiment_dict = sianalyzer.polarity_scores(msg)
                neg_sent += sentiment_dict['neg']
                neu_sent += sentiment_dict['neu']
                pos_sent += sentiment_dict['pos']
                com_sent += sentiment_dict['compound']

        if use_sentiment:
            new_data["Sentiment Negative"].append(float(neg_sent)/float(msg_count))
            new_data["Sentiment Neutral"].append(float(neu_sent) / float(msg_count))
            new_data["Sentiment Positive"].append(float(pos_sent) / float(msg_count))
            new_data["Sentiment Compound"].append(float(com_sent) / float(msg_count))


        wordset += total_words

        sim_data = ' '.join(total_words)
        new_data["words"].append(sim_data)
        if count <= 14:
            new_data["category"].append(1)

            for score, sim_num in zip(sim['SA'], sim["Simulation"]):
                sim_num = int(sim_num)
                if not math.isnan(score):
                    score = int(score)
                    if sim_num == 0:
                        low_score0[score - 3] += 1
                    elif sim_num == 1:
                        low_score1[score - 3] += 1
                    elif sim_num == 2:
                        low_score2[score - 3] += 1
                    elif sim_num == 3:
                        low_score3[score - 3] += 1

            low_turn_taking += float(sim['Total Turns'][0])
            low_sentiment += float(com_sent) / float(msg_count)
        elif count <= 28:
            new_data["category"].append(2)

            mid_turn_taking += float(sim['Total Turns'][0])
            mid_sentiment += float(com_sent) / float(msg_count)
        else:
            new_data["category"].append(3)
            for score, sim_num in zip(sim['SA'], sim["Simulation"]):
                sim_num = int(sim_num)
                if not math.isnan(score):
                    score = int(score)
                    if sim_num == 0:
                        high_score0[score - 3] += 1
                    elif sim_num == 1:
                        high_score1[score - 3] += 1
                    elif sim_num == 2:
                        high_score2[score - 3] += 1
                    elif sim_num == 3:
                        high_score3[score - 3] += 1

            high_turn_taking += float(sim['Total Turns'][0])
            high_sentiment += float(com_sent) / float(msg_count)

        new_data["SA Average"].append(sorted_scores[name])
        new_data["Total Turns"].append(float(sim['Total Turns'][0]))
        freq0, freq1, freq2, freq3 = message_frequency(sim)
        new_data["Message Frequency Sim 0"].append(freq0)
        new_data["Message Frequency Sim 1"].append(freq1)
        new_data["Message Frequency Sim 2"].append(freq2)
        new_data["Message Frequency Sim 3"].append(freq3)

    wordset_pos = nltk.pos_tag(wordset)
    pos_total_count = Counter(tag for _, tag in wordset_pos)
    for pos in pos_total_count.keys():
        new_data[pos] = []
    for cur_words, message_count in zip(new_data["words"], new_data["Total Turns"]):
        words_tokenized = word_tokenize(cur_words)
        words_pos = nltk.pos_tag(words_tokenized)
        words_pos_count = Counter(tag for _, tag in words_pos)
        for pos in pos_total_count.keys():
            if pos in words_pos_count.keys():
                count = words_pos_count[pos]
                new_data[pos].append(float(count) / float(message_count))
            else:
                new_data[pos].append(0)

    # create graphs

    score_types = (
        "SA Score 3",
        "SA Score 4",
        "SA Score 5",
        "SA Score 6"
    )
    score_counts = {
        "Simulation 0": np.array(low_score0),
        "Simulation 1": np.array(low_score1),
        "Simulation 2": np.array(low_score2),
        "Simulation 3": np.array(low_score3),
    }
    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(4)

    for simulation, score_count in score_counts.items():
        p = ax.bar(score_types, score_count, width, label=simulation, bottom=bottom)
        bottom += score_count

    ax.set_title("SA Scores of Low Performance Teams")
    ax.legend(loc="upper right")

    plt.show()

    score_counts = {
        "Simulation 0": np.array(high_score0),
        "Simulation 1": np.array(high_score1),
        "Simulation 2": np.array(high_score2),
        "Simulation 3": np.array(high_score3),
    }

    fig, ax = plt.subplots()
    bottom = np.zeros(4)

    for simulation, score_count in score_counts.items():
        p = ax.bar(score_types, score_count, width, label=simulation, bottom=bottom)
        bottom += score_count

    ax.set_title("SA Scores of High Performance Teams")
    ax.legend(loc="upper right")

    plt.show()

    low_turn_taking = low_turn_taking / 14.0
    mid_turn_taking = mid_turn_taking / 14.0
    high_turn_taking = high_turn_taking / 14.0

    low_sentiment = low_sentiment / 14.0
    mid_sentiment = mid_sentiment / 14.0
    high_sentiment = high_sentiment / 14.0

    class_labels = ["Low Performance Teams", "Mid Performance Teams", "High Performance Teams"]
    averages = [low_turn_taking, mid_turn_taking, high_turn_taking]
    fig, ax = plt.subplots()

    ax.bar(class_labels, averages)
    ax.set_ylabel("Average total messages")
    ax.set_title("Average Messages by Team Performance")
    plt.show()

    averages = [low_sentiment, mid_sentiment, high_sentiment]

    fig, ax = plt.subplots()

    ax.bar(class_labels, averages)
    ax.set_ylabel("Average Compound Sentiment")
    ax.set_title("Average Compound Sentiment by Team Performance")
    plt.show()

    clean_categorized_df = pd.DataFrame(new_data, index=sorted_scores.keys())

    clean_categorized_df.to_csv("CategorizedTeamData.csv")

    with open(pickle_file2, "wb") as categorized_pickle:
        pickle.dump(clean_categorized_df, categorized_pickle, protocol=pickle.HIGHEST_PROTOCOL)
        print("categorized dataframe pickled")

# displays graphs of stats and creates pickled dataframe of all individuals and their features and performance category
def categorize_data_person(pickle_file1, pickle_file2, use_sentiment):
    with open(pickle_file1, 'rb') as dataset_pickle:
        data = pickle.load(dataset_pickle)

    sim_names = list(data.keys())

    sorted_scores = {}

    for name in sim_names:
        for person in range(1, 5):
            sim = data[name]
            new_name = name + "-" + str(person)
            sorted_scores[new_name] = sa_average_person(sim, person)

    sorted_scores = dict(sorted(sorted_scores.items(), key=value_getter))

    count = 0
    new_data = {
        "words": [],
        "category": [],
        "SA Average": [],
        "Total Turns": [],
        "Message Frequency Sim 0": [],
        "Message Frequency Sim 1": [],
        "Message Frequency Sim 2": [],
        "Message Frequency Sim 3": []
    }

    if use_sentiment:
        new_data["Sentiment Negative"] = []
        new_data["Sentiment Neutral"] = []
        new_data["Sentiment Positive"] = []
        new_data["Sentiment Compound"] = []

    low_score0 = [0, 0, 0, 0]
    low_score1 = [0, 0, 0, 0]
    low_score2 = [0, 0, 0, 0]
    low_score3 = [0, 0, 0, 0]

    high_score0 = [0, 0, 0, 0]
    high_score1 = [0, 0, 0, 0]
    high_score2 = [0, 0, 0, 0]
    high_score3 = [0, 0, 0, 0]

    low_turn_taking = 0
    mid_turn_taking = 0
    high_turn_taking = 0

    low_sentiment = 0
    mid_sentiment = 0
    high_sentiment = 0

    wordset = []
    for name in sorted_scores.keys():
        sim_name = name[:-2]
        person = name[-1:]
        if sim_name == "341":
            test = True
        count += 1
        sim = data[sim_name]
        total_words = []
        msg_count = 0
        neg_sent = 0
        neu_sent = 0
        pos_sent = 0
        com_sent = 0
        for msg, person_num in zip(sim["Total"], sim["Turn Taking"]):
            if person == str(int(person_num)):
                msg_count += 1
                msg = str(msg)
                msg = msg.lower()
                msg_words = word_tokenize(msg)
                msg_words_lemmatized = [lemmatizer.lemmatize(word) for word in msg_words]
                stop_words = stopwords.words('english')
                all_words = words.words()
                stop_words += [".", "?", ",", "!", "\'", "..."]
                msg_words_nostop = [word for word in msg_words_lemmatized if word not in stop_words]
                msg_words_cleaned = [word for word in msg_words_nostop if word in all_words]
                total_words += msg_words_cleaned

                # add sentiment
                if use_sentiment:
                    sianalyzer = SentimentIntensityAnalyzer()
                    sentiment_dict = sianalyzer.polarity_scores(msg)
                    neg_sent += sentiment_dict['neg']
                    neu_sent += sentiment_dict['neu']
                    pos_sent += sentiment_dict['pos']
                    com_sent += sentiment_dict['compound']

        if use_sentiment:
            new_data["Sentiment Negative"].append(float(neg_sent)/float(msg_count))
            new_data["Sentiment Neutral"].append(float(neu_sent) / float(msg_count))
            new_data["Sentiment Positive"].append(float(pos_sent) / float(msg_count))
            new_data["Sentiment Compound"].append(float(com_sent) / float(msg_count))

        wordset += total_words

        sim_data = ' '.join(total_words)
        new_data["words"].append(sim_data)
        turns_name = 'Total Turns ' + str(person)
        if count <= 56:
            new_data["category"].append(1)

            for score, sim_num, person_num in zip(sim['SA'], sim["Simulation"], sim["Turn Taking"]):
                if person == str(int(person_num)):
                    sim_num = int(sim_num)
                    if not math.isnan(score):
                        score = int(score)
                        if sim_num == 0:
                            low_score0[score - 3] += 1
                        elif sim_num == 1:
                            low_score1[score - 3] += 1
                        elif sim_num == 2:
                            low_score2[score - 3] += 1
                        elif sim_num == 3:
                            low_score3[score - 3] += 1

            low_turn_taking += float(sim[turns_name][0])
            low_sentiment += float(com_sent) / float(msg_count)
        elif count <= 112:
            new_data["category"].append(2)

            mid_turn_taking += float(sim[turns_name][0])
            mid_sentiment += float(com_sent) / float(msg_count)
        else:
            new_data["category"].append(3)

            for score, sim_num, person_num in zip(sim['SA'], sim["Simulation"], sim["Turn Taking"]):
                if person == str(int(person_num)):
                    sim_num = int(sim_num)
                    if not math.isnan(score):
                        score = int(score)
                        if sim_num == 0:
                            high_score0[score - 3] += 1
                        elif sim_num == 1:
                            high_score1[score - 3] += 1
                        elif sim_num == 2:
                            high_score2[score - 3] += 1
                        elif sim_num == 3:
                            high_score3[score - 3] += 1

            high_turn_taking += float(sim[turns_name][0])
            high_sentiment += float(com_sent) / float(msg_count)

        new_data["SA Average"].append(sorted_scores[name])
        new_data["Total Turns"].append(float(sim[turns_name][0]))
        freq0, freq1, freq2, freq3 = message_frequency_person(sim, person)
        new_data["Message Frequency Sim 0"].append(freq0)
        new_data["Message Frequency Sim 1"].append(freq1)
        new_data["Message Frequency Sim 2"].append(freq2)
        new_data["Message Frequency Sim 3"].append(freq3)

    wordset_pos = nltk.pos_tag(wordset)
    pos_total_count = Counter(tag for _, tag in wordset_pos)
    for pos in pos_total_count.keys():
        new_data[pos] = []
    for cur_words, message_count in zip(new_data["words"], new_data["Total Turns"]):
        words_tokenized = word_tokenize(cur_words)
        words_pos = nltk.pos_tag(words_tokenized)
        words_pos_count = Counter(tag for _, tag in words_pos)
        for pos in pos_total_count.keys():
            if pos in words_pos_count.keys():
                count = words_pos_count[pos]
                new_data[pos].append(float(count) / float(message_count))
            else:
                new_data[pos].append(0)

    # create graphs

    score_types = (
        "SA Score 3",
        "SA Score 4",
        "SA Score 5",
        "SA Score 6"
    )
    score_counts = {
        "Simulation 0": np.array(low_score0),
        "Simulation 1": np.array(low_score1),
        "Simulation 2": np.array(low_score2),
        "Simulation 3": np.array(low_score3),
    }
    width = 0.5

    fig, ax = plt.subplots()
    bottom = np.zeros(4)

    for simulation, score_count in score_counts.items():
        p = ax.bar(score_types, score_count, width, label=simulation, bottom=bottom)
        bottom += score_count

    ax.set_title("SA Scores of Low Performance Individuals")
    ax.legend(loc="upper right")

    plt.show()

    score_counts = {
        "Simulation 0": np.array(high_score0),
        "Simulation 1": np.array(high_score1),
        "Simulation 2": np.array(high_score2),
        "Simulation 3": np.array(high_score3),
    }

    fig, ax = plt.subplots()
    bottom = np.zeros(4)

    for simulation, score_count in score_counts.items():
        p = ax.bar(score_types, score_count, width, label=simulation, bottom=bottom)
        bottom += score_count

    ax.set_title("SA Scores of High Performance Individuals")
    ax.legend(loc="upper right")

    plt.show()

    low_turn_taking = low_turn_taking / 56.0
    mid_turn_taking = mid_turn_taking / 56.0
    high_turn_taking = high_turn_taking / 56.0

    low_sentiment = low_sentiment / 56.0
    mid_sentiment = mid_sentiment / 56.0
    high_sentiment = high_sentiment / 56.0

    class_labels = ["Low Performance Individuals", "Mid Performance Individuals", "High Performance Individuals"]
    averages = [low_turn_taking, mid_turn_taking, high_turn_taking]
    fig, ax = plt.subplots()

    ax.bar(class_labels, averages)
    ax.set_ylabel("Average total messages")
    ax.set_title("Average Messages by Individual Performance")
    plt.show()

    averages = [low_sentiment, mid_sentiment, high_sentiment]
    fig, ax = plt.subplots()

    ax.bar(class_labels, averages)
    ax.set_ylabel("Average Compound Sentiment")
    ax.set_title("Average Compound Sentiment by Individual Performance")
    plt.show()

    clean_categorized_df = pd.DataFrame(new_data, index=sorted_scores.keys())

    clean_categorized_df.to_csv("CategorizedIndividualData.csv")

    with open(pickle_file2, "wb") as categorized_pickle:
        pickle.dump(clean_categorized_df, categorized_pickle, protocol=pickle.HIGHEST_PROTOCOL)
        print("categorized individual dataframe pickled")


#categorize_data("clean_data.pickle", "categorized_data.pickle", True)
#categorize_data_person("clean_data.pickle", "categorized_individual_data.pickle", True)
