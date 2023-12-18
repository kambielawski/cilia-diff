import matplotlib.pyplot as plt
import pandas as pd
from glob import glob 
import os

TRAJECTORY_COLOR = (0,0,1,0.6)

def plot_trajectory(x, y, save_path=None, show=False):
    fig, ax = plt.subplots(constrained_layout=True)

    ax.plot(x[0], y[0], '*r')
    ax.plot(x, y, color=TRAJECTORY_COLOR)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    if show:
        plt.show()

path_to_data = 'in_vitro_data/trimmed_data/*.csv'
save_path = 'in_vitro_data/plots/trajectories_trimmed/'

os.makedirs(save_path, exist_ok=True)

filenames = glob(path_to_data)

for filename in filenames:
    ID = filename.split('/')[-1].split('.')[0]
    print(ID)
    
    df = pd.read_csv(filename)

    save_filename = save_path + ID + '.png'

    # if ID=='Run4group0subject0_2':
        # plot_trajectory(df['x'], df['y'], show=True)

    plot_trajectory(df['x'], df['y'], save_path=save_filename)
