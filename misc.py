'''
Miscellaneous helper functions
'''
import numpy as np
import matplotlib.pyplot as plt
import math


def normalize(x, xmin, xmax):
    return (x-xmin)/(xmax-xmin)

# Matrix functions
def front(x,y,z,d=1):
    return x,y-d,z 

def back(x,y,z,d=1):
    return x,y+d,z

def left(x,y,z,d=1):
    return x-d,y,z

def right(x,y,z,d=1):
    return x+d,y,z

def up(x,y,z,d=1):
    return x,y,z+d

def down(x,y,z,d=1):
    return x,y,z-d

def voxcraft_to_taichi_coordinates(arr):
    return np.rot90(arr, k=1, axes=(1,2))

def taichi_to_voxcraft_coordinates(arr):
    return np.rot90(arr, k=1, axes=(2,1))

def get_n_neighbors(self, a, n=1):
    b = np.pad(a, pad_width=n, mode='constant', constant_values=0)
    neigh = np.concatenate((
        b[n*2:, n:-n, n:-n, None], b[:-n*2, n:-n, n:-n, None],
        b[n:-n, n*2:, n:-n, None], b[n:-n, :-n*2, n:-n, None],
        b[n:-n, n:-n, n*2:, None], b[n:-n, n:-n, :-n*2, None]), axis=3)
    return neigh

# Plot cilia forces
def plot_cilia_force_vectors(body, cilia, scale=5, save_dir=None, l=None, plot_multiple=False):

    if not plot_multiple: # plotting single cilia vector per voxel
        if l is not None:
            layers = [l]
        else:
            layers = range(body.shape[2])
        # Plot cilia vectors 
        for l in layers:

            layer = body[:,:,l]
            layer_cilia = cilia[:,:,l,:]
            
            if np.sum(layer==2) != 0:

                fig,ax = plt.subplots()
                ax.matshow(layer, origin='lower')

                # iterate through cells to draw on cilia vectors
                for r in range(layer_cilia.shape[0]):
                    for c in range(layer_cilia.shape[1]):
                        vector = layer_cilia[r,c,:]
                        if np.sum(vector)!=0:
                            x = vector[0]
                            y = vector[1]
                            assert vector[2]==0

                            # print(r,c)
                            # print(x,y)

                            # Draw vector onto image at r,c
                            # ax.quiver(c,r,c+x,r+y,angles='xy')
                            ax.quiver(c,r,x,y,angles=math.degrees(math.atan2(y,x)), scale=scale, pivot='tip')
                            # ax.annotate(r'({:.2f},{:.2f})'.format(x,y), (c,r))

                plt.title(f'layer {l}')
                if save_dir is not None:
                    plt.savefig(f'{save_dir}cilia_vectors_layer{l}.png')
                    plt.close()
                else:
                    plt.show()
                
    else: # for plotting a range of cilia per voxel
        assert type(cilia)==list
        cilias = cilia
        layer = body[:,:,l]
        
        fig,ax = plt.subplots()
        ax.matshow(layer, origin='lower')

        for cilia in cilias:

            layer_cilia = cilia[:,:,l,:]
            
            # iterate through cells to draw on cilia vectors
            for r in range(layer_cilia.shape[0]):
                for c in range(layer_cilia.shape[1]):
                    vector = layer_cilia[r,c,:]
                    if np.sum(vector)!=0:
                        x = vector[0]
                        y = vector[1]
                        assert vector[2]==0

                        # print(r,c)
                        # print(x,y)

                        # Draw vector onto image at r,c
                        # ax.quiver(c,r,c+x,r+y,angles='xy')
                        ax.quiver(c,r,x,y,angles=math.degrees(math.atan2(y,x)), scale=scale, pivot='tip')
                        # ax.annotate(r'({:.2f},{:.2f})'.format(x,y), (c,r))

        plt.title(f'layer {l}')

        if save_dir is not None:
            plt.savefig(save_dir)
            plt.close()
        else:
            plt.show()

def compute_eigens(cauchy_timeseries, particle=10):
    """
    Computes the eigenvalues and eigenvectors for a series of 3x3 Cauchy stress tensors.

    Parameters:
    stress_tensors (numpy.ndarray): A T x 3 x 3 array where T is the number of timesteps, and each 3x3 matrix is a Cauchy stress tensor.

    Returns:
    eig_values (numpy.ndarray): A T x 3 array of eigenvalues for each timestep.
    eig_vectors (numpy.ndarray): A T x 3 x 3 array of eigenvectors for each timestep.
    """

    stress_tensors = cauchy_timeseries[:, particle]

    T = stress_tensors.shape[0]
    eig_values = np.zeros((T, 3))
    eig_vectors = np.zeros((T, 3, 3))

    for t in range(T):
        vals, vecs = np.linalg.eig(stress_tensors[t])
        eig_values[t] = vals
        eig_vectors[t] = vecs

    return eig_values, eig_vectors

def norm_vector(v):
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return v
    return v / v_norm

def surface_normal(all_positions, particle_id, t):
    pos_t = all_positions[t, particle_id]
    centroid = np.mean(all_positions[t, :, :], axis=0)
    
    norm_vec = pos_t - centroid
    normalized_norm_vec = norm_vec / np.linalg.norm(norm_vec)

    return normalized_norm_vec

