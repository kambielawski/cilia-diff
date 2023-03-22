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

