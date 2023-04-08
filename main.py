# main for testing purposes atm
import pandas as pd
import sys
import numpy
from numpy import nan
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords

numpy.set_printoptions(threshold=sys.maxsize)

xls1 = pd.ExcelFile("dataset/discourse_analysis_one.xlsx")
xls2 = pd.ExcelFile("dataset/discourse_analysis_two.xlsx")

sims_1 = [pd.read_excel(xls1, sheet, header=None) for sheet in xls1.sheet_names]
sims_1.pop(0) # remove the metadata sheet

sims_2 = [pd.read_excel(xls2, sheet) for sheet in xls2.sheet_names]
sims_2.pop(0) # remote the metadata sheet

sims = sims_1 + sims_2
sim_names = xls1.sheet_names[1:] + xls2.sheet_names[1:]


def clean_sim_data(sim_df):
    sim_df = sim_df.rename(columns={0: "Turn Taking", 1: "Time", 2: "Total", 3: "Simulation", 4: "SA Team"})
    sim_df = sim_df[sim_df["Turn Taking"].notna()]
    sim_df = sim_df.drop(columns=sim_df.columns.values[5:])
    sim_df['SA Team'] = sim_df['SA Team'].apply(lambda x: eval("[" + str(x) + "]"))

    try:
        # If there are multiple SA, split them into two columns
        sa = sim_df.iloc[:, 4].apply(lambda x: pd.Series(x).dropna()).merge(sim_df, left_index=True, right_index=True)

        # Drop all the rows that only have one SA rating or are NAN
        sa_two = sa[sa[1].notna()].drop(columns=[0]).rename(columns={1: "SA"})
        sim_df = sa.drop(columns=[1]).rename(columns={0: "SA"})

        # Merge the copies back into one DataFrame
        sim_df = pd.concat([sim_df, sa_two])
    except:
        # If there's no item with multiple SA values in the trial, it'll throw an error, so move on
        sim_df['SA'] = sim_df['SA Team'].apply(lambda x: eval(str(x))[0])

    sim_df = sim_df.drop(columns=["SA Team"])

    if not sim_df.index.is_unique:
        # print(sim_df.index.duplicated())
        # input()
        sim_df = sim_df.set_index(pd.Series(range(0, len(sim_df))))

    return sim_df


clean_sims = {}

for i in range(len(sims)):
    try:
        clean_sims[sim_names[i]] = clean_sim_data(sims[i])
    except Exception as e:
        print(i, sim_names[i], e)
        print(sims[i])

stop_words = set(stopwords.words("english"))

f = open("viewtest.csv", 'w')
f.write("filtered words, score\n")
line_no = 0
for line in clean_sims['435']["Total"]:
    words = word_tokenize(line)
    for w in words:
        if w not in stop_words and ',' not in w:
            f.write(w + " ")
    f.write(", " + str(clean_sims['435']['SA'][line_no]) + "\n")
    line_no += 1

f.close()

print("done")

