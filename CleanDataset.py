# for removing metadata, naming columns, and splitting up multiple SA scores
import pandas as pd
import pickle
from numpy import nan

def clean_sim_data(sim_df):
    sim_df = sim_df.rename(columns={0: "Turn Taking", 1: "Time", 2: "Total", 3: "Simulation", 4: "SA Team"})
    sim_df = sim_df[sim_df["Turn Taking"].notna()]
    sim_df = sim_df.drop(columns=sim_df.columns.values[5:])
    sim_df['SA Team'] = sim_df['SA Team'].apply(lambda x: eval("[" + str(x) + "]"))
    total_turns = len(sim_df.index)
    total_turns1 = 0
    total_turns2 = 0
    total_turns3 = 0
    total_turns4 = 0
    for person in sim_df["Turn Taking"]:
        if person == 1:
            total_turns1 += 1
        elif person == 2:
            total_turns2 += 1
        elif person == 3:
            total_turns3 += 1
        elif person == 4:
            total_turns4 += 1

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

    turns = []
    turns1 = []
    turns2 = []
    turns3 = []
    turns4 = []
    for _ in sim_df["SA"]:
        turns.append(total_turns)
        turns1.append(total_turns1)
        turns2.append(total_turns2)
        turns3.append(total_turns3)
        turns4.append(total_turns4)
    sim_df = sim_df.drop(columns=["SA Team"])
    sim_df['Total Turns'] = turns
    sim_df['Total Turns 1'] = turns1
    sim_df['Total Turns 2'] = turns2
    sim_df['Total Turns 3'] = turns3
    sim_df['Total Turns 4'] = turns4

    if not sim_df.index.is_unique:
        # print(sim_df.index.duplicated())
        # input()
        sim_df = sim_df.set_index(pd.Series(range(0, len(sim_df))))

    return sim_df

def clean_dataset():
    xls1 = pd.ExcelFile("dataset/discourse_analysis_one.xlsx")
    xls2 = pd.ExcelFile("dataset/discourse_analysis_two.xlsx")

    sims_1 = [pd.read_excel(xls1, sheet, header=None) for sheet in xls1.sheet_names]
    sims_1.pop(0) # remove the metadata sheet

    sims_2 = [pd.read_excel(xls2, sheet) for sheet in xls2.sheet_names]
    sims_2.pop(0) # remote the metadata sheet

    sims = sims_1 + sims_2
    sim_names = xls1.sheet_names[1:] + xls2.sheet_names[1:]

    clean_sims = {}

    for i in range(len(sims)):
        try:
            clean_sims[sim_names[i]] = clean_sim_data(sims[i])
            csv_name = "CleanedTeamCSVs\\" + sim_names[i] + ".csv"
            clean_sims[sim_names[i]].to_csv(csv_name)
        except Exception as e:
            print(i, sim_names[i], e)
            print(sims[i])

    with open("clean_data.pickle", "wb") as pickle_file:
        pickle.dump(clean_sims, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        print("Dataset Cleaned")

#clean_dataset()
