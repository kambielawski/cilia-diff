import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import pickle
import os

def split(df, frames_per_chunk):
    # split dataframe into 30 sec chunks
    # returns list of dataframes

    number_of_frames = len(df)

    curr_frames_left = number_of_frames

    if number_of_frames<=frames_per_chunk:
        return [df]
    
    chunks = []
    while curr_frames_left>0:
        start = number_of_frames-curr_frames_left#+1

        stop = start+frames_per_chunk

        if stop>number_of_frames: # the last chunk might be smaller
            stop=start+curr_frames_left

        chunk = df.iloc[start:stop].reset_index()

        if not is_chunk_missing_frames(chunk): # do not include if there is a frame missing in the chunk
            chunks.append(chunk)
        
        curr_frames_left-=(stop-start)
    
    return chunks

def is_chunk_missing_frames(chunk):
    # Checks frame # in chunks to make sure they are all consecutive
    # Returns True is frames are missing from the check and False otherwise

    frames = list(chunk['frame'])
    frames_missing = False

    for i in range(len(frames)-1):
        if frames[i] + 1 != frames[i+1]: # frames are consecutive
            frames_missing = True

    return frames_missing
 
frame_rate = 0.4 #fps
CHUNK_LENGTH = 30 # seconds
frames_per_chunk = int(frame_rate*CHUNK_LENGTH )

print("Frames per chunk:", frames_per_chunk)

os.makedirs("in_vitro_data/split_data/", exist_ok=True)

#############################################################
# Remove this chunk and decrease indent to process a single bot

BOT_IDs=[]
for bot_csv in glob('in_vitro_data/trimmed_data/*.csv'):
    strp = bot_csv.split('/')[-1]
    if '_' in strp:
        BOT_ID = strp.split('_')[0]
    else:
        BOT_ID = strp.split('.')[0]
    
    if BOT_ID not in BOT_IDs:
        BOT_IDs.append(BOT_ID)

for BOT_ID in BOT_IDs:
#############################################################
    os.makedirs("in_vitro_data/split_data/{}_{}sec_segments/".format(BOT_ID, CHUNK_LENGTH), exist_ok=True)

    run = 0
    for i,bot_csv in enumerate(glob('in_vitro_data/trimmed_data/{}*.csv'.format(BOT_ID))):

        print(bot_csv)

        # Read in CSV and clean
        df = pd.read_csv(bot_csv)

        # Split into chunks
        # Frame rate = 40 fps
        trajectories = split(df, frames_per_chunk)

        # Save out trajectories
        for j,trajectory in enumerate(trajectories):
            
            csv_save_path = "in_vitro_data/split_data/{}_{}sec_segments/{}_run{}.csv".format(BOT_ID, CHUNK_LENGTH, BOT_ID, run)
            trajectory.to_csv(csv_save_path)

            run+=1


        
    
