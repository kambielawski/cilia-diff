from bot import Bot
import pickle
import numpy as np
import misc
import matplotlib.pyplot as plt

TEST_BOT_PKL_FILE_PATH = './pickle/circular/Run4group3subject2/Run4group3subject2_res38.p'

anthrobot = Bot(TEST_BOT_PKL_FILE_PATH)

def cross_section_visualize(body):
    w,l,h = body.shape
    for x in range(w):
        yz_points_body = []
        yz_points_act = []
        for y in range(l):
            for z in range(h):
                if body_arr[x][y][z] == 1: 
                    yz_points_body.append((y, z))  
                if body_arr[x][y][z] == 2: 
                    yz_points_act.append((y, z))
                    # yz_points.append((y,z))
        if len(yz_points_body):
            plt.scatter(*zip(*yz_points_body), label="body")
        if len(yz_points_act):
            plt.scatter(*zip(*yz_points_act), label="act")
        # plt.scatter(*zip(*yz_points))
        plt.legend()
        plt.show()

def main():
    body_arr = np.array(anthrobot.body)

    print(np.sum(body_arr > 0))

    # Now we need to create the particles for each voxel
    # If a voxel is ciliated, the particle is an actuator
    

        
    

if __name__ == '__main__':
    main()
