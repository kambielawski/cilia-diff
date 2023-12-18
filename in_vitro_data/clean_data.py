import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import pickle
import os


def clean(df, ID_NBR, filter_by_ID_nbr=True):
    # filters out ignore=True rows
    # track fixed column is the BOT ID 
    clean_df = df[df['ignore']==False].reset_index(drop=True)

    if ID_NBR is not None:
        clean_df = clean_df[clean_df['track_fixed']==int(ID_NBR)].reset_index(drop=True)
    
    return clean_df


os.makedirs("in_vitro_data/cleaned_data/",exist_ok=True)

#############################################################
# Remove this chunk and decrease indent to process a single bot
BOT_IDs=[]
for bot_csv in glob('in_vitro_data/raw_data/*.csv'):
    strp = bot_csv.split('/')[-1]
    if '_' in strp:
        BOT_ID = strp.split('_')[0]
    else:
        BOT_ID = strp.split('.')[0]
    
    if BOT_ID not in BOT_IDs:
        BOT_IDs.append(BOT_ID)

for BOT_ID in BOT_IDs:
#############################################################
    
    for i,bot_csv in enumerate(glob('in_vitro_data/raw_data/{}*.csv'.format(BOT_ID))):

        print(bot_csv)

        if bot_csv.split('/')[-1] == 'Run8bot2_2.csv': # skip this CSV for now until discussing with Gizem
            continue

        # Read in CSV and clean
        df_raw = pd.read_csv(bot_csv)

        # Only filter by ID_NBR for circulars
        if 'subject' in BOT_ID:
            ID_NBR = BOT_ID.split('subject')[-1]
        else:
            ID_NBR = None

        # Problem CSVs
        problem_csvs = ['Run6group5subject5.csv', 'Run4group3subject2.csv', 'Run6group5subject1_2.csv']

        filename = bot_csv.split('/')[-1]
        if filename in problem_csvs:
            ID_NBR=None # ignore the bot ID because it doesn't match the filename but has been manually checked by Gizem

        if "ignore" in df_raw.columns:
            df = clean(df_raw, ID_NBR)
        else:
            df = df_raw
            
        csv_save_path = "in_vitro_data/cleaned_data/{}".format(bot_csv.split('/')[-1])
        df.to_csv(csv_save_path)
