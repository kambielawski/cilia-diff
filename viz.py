import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import scipy

from misc import taichi_to_voxcraft_coordinates, surface_normal

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

def vis_cauchy_matrix_diag(matrix_timeseries, particle=0):
    time = range(len(matrix_timeseries))

    print(matrix_timeseries[0, particle]) # Should be 3x3

    xx_data = matrix_timeseries[:, particle, 0, 0]
    yy_data = matrix_timeseries[:, particle, 1, 1]
    zz_data = matrix_timeseries[:, particle, 2, 2]

    plt.figure(figsize=(12, 8))

    # Plot x vs time
    plt.subplot(3, 1, 1)
    plt.plot(time, xx_data, label='Cauchy XX', color='red')
    plt.title('Cauchy XX Component vs Time')
    plt.xlabel('Time')
    plt.ylabel('XX Component')

    # Plot y vs time
    plt.subplot(3, 1, 2)
    plt.plot(time, yy_data, label='Cauchy YY', color='green')
    plt.title('Cauchy YY Component vs Time')
    plt.xlabel('Time')
    plt.ylabel('YY Component')

    # Plot z vs time
    plt.subplot(3, 1, 3)
    plt.plot(time, zz_data, label='Cauchy ZZ', color='blue')
    plt.title('Cauchy ZZ Component vs Time')
    plt.xlabel('Time')
    plt.ylabel('ZZ Component')

    plt.tight_layout()
    plt.show()

def vis_cauchy_matrix(matrix_timeseries, particle=0):
    time = range(len(matrix_timeseries))

    print(matrix_timeseries[0, particle]) # Should be 3x3

    particle_timeseries = matrix_timeseries[:, particle]

    component_label_map = [
        ['{XX}', '{XY}', '{XZ}'],
        ['{YX}', '{YY}', '{YZ}'],
        ['{ZX}', '{ZY}', '{ZZ}']
    ]
    yax_min = np.min(particle_timeseries)
    yax_max = np.max(particle_timeseries)

    T, rows, cols = particle_timeseries.shape
    if rows != 3 or cols != 3:
        raise ValueError("The matrix must be of shape T x 3 x 3")

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    # Hide the lower left 3 subplots
    for i in range(3):
        for j in range(i):
            axs[i, j].axis('off')

    # Plot only the upper triangular part of the matrix
    time = np.arange(T)
    for i in range(rows):
        for j in range(i, cols):
            fft_result = np.fft.fft(particle_timeseries[:, i, j])
            power_spectrum = np.abs(fft_result)**2
            energy = np.sum(power_spectrum)

            axs[i, j].plot(time, particle_timeseries[:, i, j])
            axs[i, j].set_ylim((yax_min, yax_max))
            axs[i, j].set_title(fr'$\sigma_{component_label_map[i][j]}$ Component') # , Energy: {energy:.2f}')
            axs[i, j].set_xlabel('Timestep')
            axs[i, j].set_ylabel('Value')

    plt.tight_layout()
    plt.show()

# def vis_cauchy_eigenvectors(matrix_timeseries, particle=0):
#     particle_timeseries = matrix_timeseries[:, particle]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def animate_principal_stress(eig_values, eig_vectors):
    """
    Animates the first principal stress vector in 3D over T timesteps.

    Parameters:
    eig_values (numpy.ndarray): A T x 3 array of eigenvalues for each timestep.
    eig_vectors (numpy.ndarray): A T x 3 x 3 array of eigenvectors for each timestep.
    T (int): Number of timesteps.
    """
    
    # Initialize the figure for animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('First Principal Stress Vector Animation in 3D')

    # Plotting the largest eigenvector
    quiver = ax.quiver(0, 0, 0, 0, 0, 0, color='r')
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

    def update(frame):
        # Select the largest eigenvalue and its corresponding eigenvector
        max_eigenvalue_index = np.argmax(eig_values[frame])
        vector = eig_vectors[frame, :, max_eigenvalue_index]
        quiver.set_segments([[(0, 0, 0), tuple(vector)]])
        time_text.set_text('Timestep: {}'.format(frame))
        return quiver,

    # Creating the animation
    anim = FuncAnimation(fig, update, frames=eig_values.shape[0], interval=20, blit=False)
    anim.save('eigenvector.gif', writer='imagemagick')

    return anim

def plot_max_eigenvalue_over_time(cauchy_timeseries, particle=10):
    """
    Plots the maximum eigenvalue of a series of 3x3 Cauchy stress tensors over time.

    Parameters:
    stress_tensors (numpy.ndarray): A T x 3 x 3 array where T is the number of timesteps, and each 3x3 matrix is a Cauchy stress tensor.
    """

    stress_tensors = cauchy_timeseries[:, particle]
    T = stress_tensors.shape[0]
    max_eigenvalues = np.zeros(T)

    for t in range(T):
        eigenvalues = np.linalg.eigvals(stress_tensors[t])
        max_eigenvalues[t] = np.max(eigenvalues.real)  # Using real part in case of complex numbers

    # Plotting
    plt.plot(range(T), max_eigenvalues, label='Max Eigenvalue')
    plt.xlabel('Timestep')
    plt.ylabel('Maximum Eigenvalue')
    plt.title('Maximum Eigenvalue of Cauchy Stress Tensor Over Time')
    plt.legend()
    plt.show()

def ensure_consistency_eigenvectors(eigenvectors):
    for i in range(1, len(eigenvectors)):
        if np.dot(eigenvectors[i], eigenvectors[i-1]) < 0:
            eigenvectors[i] *= -1
    return eigenvectors

def plot_principal_stress_component_over_time(cauchy_timeseries, particle=10, component='x'):
    """
    Plots the specified component (x, y, or z) of the first principal stress vector over time.

    Parameters:
    stress_tensors (numpy.ndarray): A T x 3 x 3 array where T is the number of timesteps, and each 3x3 matrix is a Cauchy stress tensor.
    component (str): The component to plot ('x', 'y', or 'z').
    """
    stress_tensors = cauchy_timeseries[:, particle]
    T = stress_tensors.shape[0]
    component_values = np.zeros(T)
    component_index = {'x': 0, 'y': 1, 'z': 2}.get(component, 0)  # Defaults to 'x' if invalid component is given

    principal_vector_over_time = []

    for t in range(T):
        eigenvalues, eigenvectors = np.linalg.eig(stress_tensors[t])
        # eigenvalues_sp, eigenvectors_sp_left, eigenvectors_sp_right = scipy.linalg.eig(stress_tensors[t], left=True)
        if t == 10:
            print(eigenvalues, eigenvectors)
        max_eigenvalue_index = np.argmax(eigenvalues.real)  # Using real part in case of complex numbers
        principal_vector = eigenvectors[:, max_eigenvalue_index].real  # Using real part
        principal_vector_over_time.append(principal_vector)
        if t > 0:
            if np.dot(principal_vector_over_time[t], principal_vector_over_time[t-1]) < 0:
                principal_vector_over_time[t] *= -1
        component_values[t] = principal_vector_over_time[t][component_index]

    # Plotting
    plt.plot(range(T), component_values, label=f'{component.upper()} Component of First Principal Stress Vector')
    plt.xlabel('Timestep')
    plt.ylabel(f'{component.upper()} Component Value')
    plt.title(f'{component.upper()} Component of First Principal Stress Vector Over Time')
    plt.legend()
    plt.show()



def norm_stress_angle_plot(act_timeseries, pos_timeseries, particle_id, actuator_id):
    # particle_timeseries = act_timeseries[:, particle]
    T = act_timeseries.shape[0]

    angles = np.zeros(T)
    traction_values = np.zeros((T,3))

    for timestep in range(T):
        cauchy_matrix = act_timeseries[timestep, actuator_id]
        normal_vec = surface_normal(pos_timeseries, particle_id, t=timestep)

        traction_vec = np.dot(cauchy_matrix, normal_vec) / np.linalg.norm(np.dot(cauchy_matrix, normal_vec))
        traction_values[timestep] = traction_vec

        if timestep > 0 and np.dot(traction_values[timestep], traction_values[timestep-1]) < 0:
            traction_values[timestep] *= -1

        dot_product = np.dot(normal_vec, traction_values[timestep])
        # Compute the angle in radians
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        angles[timestep] = angle_deg

    plt.plot(range(T), angles)
    plt.xlabel('Timestep')
    plt.ylabel(f'Actuator {actuator_id} Angle')
    plt.title(f'Actuator {actuator_id} Traction-Normal Angle Over Time')
    plt.legend()
    plt.show()



    # for t in particle_timeseries:
        # normal_vec = normal_vector(cauchy_timeseries, particle, )

def traction_vec_plot(act_timeseries, pos_timeseries, particle_id, actuator_id):

    T = act_timeseries.shape[0]

    traction_values = np.zeros((T,3))

    for timestep in range(T):
        cauchy_matrix = act_timeseries[timestep, actuator_id]
        normal_vec = surface_normal(pos_timeseries, particle_id, t=timestep)

        traction_vec = np.dot(cauchy_matrix, normal_vec) / np.linalg.norm(np.dot(cauchy_matrix, normal_vec))

        traction_values[timestep] = traction_vec
    
    traction_values = ensure_consistency_eigenvectors(traction_values)

    plt.plot(range(T), traction_values[:, 0], label='X component')
    plt.plot(range(T), traction_values[:, 1], label='Y component')
    plt.plot(range(T), traction_values[:, 2], label='Z component')
    plt.xlabel('Timestep')
    plt.ylabel(f'Traction Vector Values')
    plt.title(f'Traction Vec over time')
    plt.legend()
    plt.show()


def normal_vec_plot(act_timeseries, pos_timeseries, particle_id, actuator_id):

    T = act_timeseries.shape[0]

    normal_vec_values = np.zeros((T, 3))

    for timestep in range(T):
        normal_vec = surface_normal(pos_timeseries, particle_id, t=timestep)

        normal_vec_values[timestep] = normal_vec
    
    plt.plot(range(T), normal_vec_values[:, 0], label='X component')
    plt.plot(range(T), normal_vec_values[:, 1], label='Y component')
    plt.plot(range(T), normal_vec_values[:, 2], label='Z component')
    plt.xlabel('Timestep')
    plt.ylabel(f'Normal vec values')
    plt.title(f'Normal Vec over time')
    plt.legend()
    plt.show()

def norm_stress_angle_distribution_plot(act_timeseries, pos_timeseries, actuator_ids):
    # particle_timeseries = act_timeseries[:, particle]
    T = act_timeseries.shape[0]
    n_actuators = act_timeseries.shape[1]

    angles = np.zeros(T)
    traction_values = np.zeros((T,3))

    mean_angles = np.zeros(n_actuators)

    for actuator_id in range(n_actuators):
        for timestep in range(T):
            particle_id = np.where(actuator_ids == actuator_id)[0][0]
            cauchy_matrix = act_timeseries[timestep, actuator_id]
            normal_vec = surface_normal(pos_timeseries, particle_id, t=timestep)

            traction_vec = np.dot(cauchy_matrix, normal_vec) / np.linalg.norm(np.dot(cauchy_matrix, normal_vec))
            traction_values[timestep] = traction_vec

            if timestep > 0 and np.dot(traction_values[timestep], traction_values[timestep-1]) < 0:
                traction_values[timestep] *= -1

            dot_product = np.dot(normal_vec, traction_values[timestep])
            # Compute the angle in radians
            angle_rad = np.arccos(dot_product)
            angle_deg = np.degrees(angle_rad)
            angles[timestep] = angle_deg

        mean_angles[actuator_id] = np.mean(angles)
        if actuator_id % 20 == 0:
            print(f'{actuator_id+1}/{n_actuators} actuators complete')
            print(f'Number of nans: ')

    plt.hist(mean_angles, bins='auto')
    plt.xlabel('Timestep')
    plt.ylabel(f'Actuator {actuator_id} Angle')
    plt.title(f'Actuator {actuator_id} Traction-Normal Angle Over Time')
    plt.legend()
    plt.show()

def anthrobot_angle_3dplot(act_timeseries, pos_timeseries, actuator_ids):
    # particle_timeseries = act_timeseries[:, particle]
    T = act_timeseries.shape[0]
    n_actuators = act_timeseries.shape[1]

    # Actuator position timeseries
    actuator_position_timeseries = np.zeros((n_actuators, 3))
    for i, particle in enumerate(pos_timeseries[1]): # iterate over particles
        if actuator_ids[i] != -1:
            actuator_position_timeseries[actuator_ids[i]] = particle

    angles = np.zeros(T)
    traction_values = np.zeros((T,3))
    mean_angles = np.zeros(n_actuators)

    for actuator_id in range(n_actuators):
        for timestep in range(T):
            particle_id = np.where(actuator_ids == actuator_id)[0][0]
            cauchy_matrix = act_timeseries[timestep, actuator_id]
            normal_vec = surface_normal(pos_timeseries, particle_id, t=timestep)

            traction_vec = np.dot(cauchy_matrix, normal_vec) / np.linalg.norm(np.dot(cauchy_matrix, normal_vec))
            traction_values[timestep] = traction_vec

            if timestep > 0 and np.dot(traction_values[timestep], traction_values[timestep-1]) < 0:
                traction_values[timestep] *= -1

            dot_product = np.dot(normal_vec, traction_values[timestep])
            # Compute the angle in radians
            angle_rad = np.arccos(dot_product)
            angle_deg = np.degrees(angle_rad)
            angles[timestep] = angle_deg

        mean_angles[actuator_id] = np.mean(angles)
        if actuator_id % 50 == 0:
            print(f'{actuator_id}/{n_actuators} actuators complete')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a scatter plot
    scatter = ax.scatter(actuator_position_timeseries[:, 0], actuator_position_timeseries[:, 1], actuator_position_timeseries[:, 2], c=mean_angles, cmap='twilight_shifted')

    # Adding a color bar which maps values to colors
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Normal Traction Angle')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    plt.show()