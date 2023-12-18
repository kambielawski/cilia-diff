import matplotlib.pyplot as plt
import pandas as pd
from glob import glob 
import os

TRAJECTORY_COLOR = (0,0,1,0.6)

path_to_data = 'in_vitro_data/split_data/*'
save_path = 'in_vitro_data/plots/trajectories_split_raw/'

os.makedirs(save_path, exist_ok=True)

folders = glob(path_to_data)

for folder in folders:
    ID = folder.split('/')[-1]
    
    fig, ax = plt.subplots(constrained_layout=True)

    for filename in glob(folder+'/*.csv'):
    
        df = pd.read_csv(filename)
        
        # Raw 
        ax.plot(df['x'][0], df['y'][0], '*r') 
        ax.plot(df['x'], df['y'], color=TRAJECTORY_COLOR)

        # # For shifted trajectories (starting at (0,0))
        # ax.plot(df['x_shift'][0], df['y_shift'][0], '*r') 
        # ax.plot(df['x_shift'], df['y_shift'], color=TRAJECTORY_COLOR)
        
        # For shifted and rotated trajectories
        # ax.plot(df['x_rotate'][0], df['y_rotate'][0], '*r') 
        # ax.plot(df['x_rotate'], df['y_rotate'], color=TRAJECTORY_COLOR)

    save_filename = save_path + ID + '_raw.png'
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close()
