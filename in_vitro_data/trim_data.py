from importlib.resources import path
import os, sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import constants


# trim dictionary - key = csv, value = (start, stop) in seconds
# stop value of -1 indicates end of video
# trim values obtained from: https://docs.google.com/document/d/11swa_ZKFVBN_bmgecWx4JJk74FSpRO541HZx_oK3cFY/edit#heading=h.ujr91xlpxt19

trim_dict = {'Run4group0subject0_1': (15,120), 'Run5group7subject1':(0,-1),\
            'Run4group3subject2': (0,-1), 'Run4group0subject0_2':(0,-1), \
                'Run4group0subject5': (8,-1), 'Run6group5subject1_1':(0,-1),\
                    'Run6group5subject1_2':(0,-1), 'Run6group5subject2': (10,-1),\
                        'Run6group5subject4':(0,-1), 'Run6group5subject5':(0,95), \
                            'Run8bot1_1':(2,13),'Run8bot1_2':(0,-1), 'Run8bot1_3':(0,-1),\
                                'Run8bot2':(0,-1), 'Run8bot4':(0,-1), 'Run8bot5':(0,-1),\
                                    'Run8bot6':(0,-1), 'Run8bot7':(0,-1), 'Run8bot9':(0,-1),\
                                        'Run8bot13':(0,-1), 'Run4group0subject4_1':(0,-1), 'Run4group0subject4_3':(0,-1),\
                                            'Run6group5subject3':(64,110)}

video_length_dict = {'Run4group0subject0_1': 129, 'Run5group7subject1':23,\
            'Run4group3subject2': 155, 'Run4group0subject0_2': 34, \
                'Run4group0subject5': 222, 'Run6group5subject1_1': 17,\
                    'Run6group5subject1_2':3, 'Run6group5subject2': 60,\
                        'Run6group5subject4': 48, 'Run6group5subject5': 136, \
                            'Run8bot1_1':4,'Run8bot1_2': 6, 'Run8bot1_3':7,\
                                'Run8bot2':3, 'Run8bot4':4, 'Run8bot5':9,\
                                    'Run8bot6':4, 'Run8bot7':5, 'Run8bot9':4,\
                                        'Run8bot13':3, 'Run4group0subject4_1':15, 'Run4group0subject4_3':4,\
                                            'Run6group5subject3':140}                                    

path_to_data = 'in_vitro_data/cleaned_data/'
save_path = 'in_vitro_data/trimmed_data/'
os.makedirs(save_path, exist_ok=True)

for BOT_ID in trim_dict:
    print(BOT_ID)

    df = pd.read_csv(path_to_data + BOT_ID + '.csv')

    # Find total length of video in real-time (number of frames * frame rate)

    first_frame = df['frame'][0]
    last_frame = df['frame'][len(df)-1]
    total_number_frames = last_frame-first_frame
    
    real_length = total_number_frames * (1/constants.frame_rate) # in seconds
    video_length = video_length_dict[BOT_ID]

    speedup_factor = real_length/video_length # should be about 40 (1 video sec = ~ 40 real sec)
    print(speedup_factor)

    # Calculate how many frames to cut out based on the frame rate and start/stop
    start,stop = trim_dict[BOT_ID]
 
    start_frame = int(np.floor((start*speedup_factor) * constants.frame_rate))
    end_frame = int(np.floor((stop*speedup_factor) * constants.frame_rate))

    print(start_frame, end_frame)

    df.drop(labels=[df.columns[0]], axis=1, inplace=True)     

    if start != 0: # trim from beginning
        
        df = df[df['frame']>start_frame]
    
    if stop != -1: # trim from end

        df = df[df['frame']<end_frame]
    
    # Save trajectories out
    df.to_csv(save_path + BOT_ID + '.csv')


