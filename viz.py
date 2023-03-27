import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


'''
data = np.array([[[1,2,3],[1,3,3],[1,4,3]],
                 [[1,2,3],[1,3,3],[1,4,3]],
                 [[1,2,3],[1,3,3],[1,4,3]]])
'''


def visualize(time_series):
    timesteps, n_particles, dim = time_series.shape 

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set plot limits (optional)
    ax.set_xlim(np.min(time_series), np.max(time_series))
    ax.set_ylim(np.min(time_series), np.max(time_series))
    ax.set_zlim(np.min(time_series), np.max(time_series))

    def update_scatter(num):
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(np.min(time_series), np.max(time_series))
        ax.set_ylim(np.min(time_series), np.max(time_series))
        ax.set_zlim(np.min(time_series), np.max(time_series))
        ax.scatter(time_series[num, :, 0], time_series[num, :, 1], time_series[num, :, 2])

        timestep_label = f'Timestep: {num}'
        legend = ax.legend([timestep_label], handlelength=0, handletextpad=0)
        for item in legend.legendHandles:
            item.set_visible(False)

    # Create the animation
    ani = animation.FuncAnimation(fig, update_scatter, frames=len(time_series), interval=100, repeat=True)
    # ani.legend()

    # Display the animation
    plt.show()

