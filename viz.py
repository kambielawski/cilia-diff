import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from misc import taichi_to_voxcraft_coordinates

'''
data = np.array([[[1,2,3],[1,3,3],[1,4,3]],
                 [[1,2,3],[1,3,3],[1,4,3]],
                 [[1,2,3],[1,3,3],[1,4,3]]])
'''

def cross_section_visualize(body):
    w,l,h = body.shape
    for x in range(w):
        yz_points_body = []
        yz_points_act = []
        for y in range(l):
            for z in range(h):
                if body[x][y][z] == 1: 
                    yz_points_body.append((y, z))  
                if body[x][y][z] == 2: 
                    yz_points_act.append((y, z))
                    # yz_points.append((y,z))
        if len(yz_points_body):
            plt.scatter(*zip(*yz_points_body), label="body")
        if len(yz_points_act):
            plt.scatter(*zip(*yz_points_act), label="act")
        # plt.scatter(*zip(*yz_points))
        plt.legend()
        plt.show()


def visualize_actuator_flat(time_series, actuator_ids):
    # time_series = time_series[::5]
    timesteps, n_particles, dim = time_series.shape

    n_actuators = np.sum(actuator_ids != -1)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111) #, projection='2d')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Set plot limits (optional)
    ax.set_xlim(np.min(time_series), np.max(time_series))
    ax.set_ylim(np.min(time_series), np.max(time_series))

    # scene.body is n * n * n matrix of 0,1,2 (1 body, 2 ciliated)

    # Separate ciliated from non-ciliated particles
    actuator_time_series = np.zeros((timesteps, n_actuators, dim))
    # body_time_series = np.zeros((timesteps, n_particles - scene.num_actuators, dim))
    for t, pos_data in enumerate(time_series):  # iterate over timesteps 
        p = 0
        for i, particle in enumerate(pos_data): # iterate over particles
            if actuator_ids[i] != -1:
                actuator_time_series[t, actuator_ids[i]] = particle
            # else:
                # body_time_series[t, p] = particle
                # p += 1


    def update_scatter(timestep):
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(np.min(time_series), np.max(time_series))
        ax.set_ylim(np.min(time_series), np.max(time_series))

        # ax.scatter(body_time_series[timestep, :, 0], body_time_series[timestep, :, 2], body_time_series[timestep, :, 1])
        ax.scatter(actuator_time_series[timestep, :, 0], actuator_time_series[timestep, :, 2]) # , actuator_time_series[timestep, :, 1])

        t = timestep * 5
        timestep_label = f'Timestep: {t}'
        legend = ax.legend([timestep_label], handlelength=0, handletextpad=0)
        for item in legend.legendHandles:
            item.set_visible(False)

    # Create the animation
    ani = animation.FuncAnimation(fig, update_scatter, frames=len(time_series), interval=100, repeat=False)
    # ani.legend()

    ani.save('animation.gif', writer='ffmpeg', fps=30, dpi=100)

    # Display the animation
    plt.show()


def visualize(actuator_id, time_series, iteration, ratio=20):
    time_series = time_series[::ratio]
    timesteps, n_particles, dim = time_series.shape
    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    min_position, max_position = np.min(time_series), np.max(time_series)

    # Set plot limits (optional)
    print(f'Animation: {timesteps} timesteps')
    ax.set_xlim(min_position, max_position)
    ax.set_ylim(min_position, max_position)
    ax.set_zlim(min_position, max_position)
    colors = ['#1f77b4' if actuator_id[i] < 0 else '#ff7f0e' for i in range(n_particles)]

    def update_scatter(timestep):
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(min_position, max_position)
        ax.set_ylim(min_position, max_position)
        ax.set_zlim(min_position, max_position)

        ax.scatter(time_series[timestep, :, 0], time_series[timestep, :, 2], time_series[timestep, :, 1], c=colors)

        # Set camera position 
        azimuth_angle = 270 
        elevation_angle = 5
        ax.view_init(elev=elevation_angle, azim=azimuth_angle)

        st = timestep * ratio
        timestep_label = f'Timestep: {st}'

        legend = ax.legend([timestep_label], handlelength=0, handletextpad=0)
        for item in legend.legendHandles:
            item.set_visible(False)

    # Create the animation
    ani = animation.FuncAnimation(fig, update_scatter, frames=len(time_series), interval=100, repeat=False)
    ani.save(f'animation_{iteration}.gif', writer='ffmpeg', fps=30, dpi=100)
    # plt.close(fig)
    # Display the animation
    plt.show()

def plot_single_particle(particle_timeseries):
    time = range(len(particle_timeseries))
    x_data = particle_timeseries[:, 0]
    y_data = particle_timeseries[:, 1]
    z_data = particle_timeseries[:, 2]

    plt.figure(figsize=(12, 8))

    # Plot x vs time
    plt.subplot(3, 1, 1)
    plt.plot(time, x_data, label='X Position', color='red')
    plt.title('X Position vs Time')
    plt.xlabel('Time')
    plt.ylabel('X Position')

    # Plot y vs time
    plt.subplot(3, 1, 2)
    plt.plot(time, y_data, label='Y Position', color='green')
    plt.title('Y Position vs Time')
    plt.xlabel('Time')
    plt.ylabel('Y Position')

    # Plot z vs time
    plt.subplot(3, 1, 3)
    plt.plot(time, z_data, label='Z Position', color='blue')
    plt.title('Z Position vs Time')
    plt.xlabel('Time')
    plt.ylabel('Z Position')

    plt.tight_layout()
    plt.show()

def plot_fft_timeseries(particle_timeseries, direction='x'):
    direction2idx = { 'x': 0, 'y': 1, 'z': 2 } 
    direction_idx = direction2idx[direction]

    values = particle_timeseries[:, direction_idx]
    values = values - np.mean(values)
    time = np.arange(len(values)) * 0.001

    # Calculate FFT
    fft_values = np.fft.fft(values)
    fft_freq = np.fft.fftfreq(len(time), (time[1] - time[0]))

    # Consider only the positive frequencies
    n = len(fft_freq) // 2
    positive_freqs = fft_freq[:n]
    positive_fft = np.abs(fft_values)[:n]

    # Find the indices of the three most prominent frequencies (excluding zero frequency)
    indices = np.argsort(positive_fft[1:])[-3:] + 1  # +1 to exclude zero frequency

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot time series
    axs[0].plot(time, values, label='Original Time Series')
    axs[0].set_title('Time Series')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Value')

    # Plot FFT
    axs[1].plot(positive_freqs, positive_fft, label='FFT of Time Series')
    for idx in indices:
        axs[1].axvline(x=positive_freqs[idx], color='r', linestyle='--')
        axs[1].text(positive_freqs[idx], positive_fft[idx], f' {positive_freqs[idx]:.2f} Hz', verticalalignment='bottom')
    axs[1].set_title('FFT of Time Series')
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('Magnitude')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()