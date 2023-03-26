'''
class for an anthrobot
'''

import numpy as np
import pickle 
from scipy.ndimage.measurements import center_of_mass
from misc import up,down,front,back,left,right, plot_cilia_force_vectors, voxcraft_to_taichi_coordinates

class Bot():

    def __init__(self, filename, make_sphere=False, ID=None, make_surface_ciliated=False):

        self.ID = ID
        
        with open(filename, 'rb') as f:
            # Rotate the body to match TaiChi coordinate system
            self.body = voxcraft_to_taichi_coordinates(pickle.load(f)[0])
            # self.body = pickle.load(f)[0]
            self.true_morph = self.body # save copy

        # Make self.body into a sphere
        if make_sphere:
            self.sphere = True
            self.make_sphere()
        else:
            self.sphere = False

        if make_surface_ciliated:
            self.ciliated_surface = True
            self.make_surface_ciliated()
        else:
            self.ciliated_surface=False

    def make_sphere(self):
        # makes body into a sphere with about the same number of cells as true morphology

        # compute radius of sphere
        n_cells = np.sum(self.true_morph>0) # volume of sphere
        radius = int(np.round(np.cbrt(n_cells*3/(4*np.pi))))

        # make sphere 
        length = radius*2+1
        body = np.zeros((length, length, length), dtype=int)
        r2 = np.arange(-radius, radius + 1) ** 2
        dist2 = r2[:, None, None] + r2[:, None] + r2
        body[dist2 < radius ** 2] = 1

        self.body = body

    def make_surface_ciliated(self):
        '''Makes all surface cells ciliated'''

        neigh = self.get_neighbors(self.body)

        for x in range(self.body.shape[0]):
            for y in range(self.body.shape[1]):
                for z in range(self.body.shape[2]):
                    if self.body[x,y,z]==1 and np.sum(neigh[x,y,z,:])<6: # surface non-ciliated cell
                        self.body[x,y,z]=2 # make ciliated cell


    def set_cilia_forces(self, randomize_distribution=False, restricted=True, align=False, align_angle=None, unit=False):
        
        # Set cilia distribution
        if randomize_distribution:
            self.randomize_cilia_distribution() # works for sphere and true morph
        elif self.sphere: # but not randomize distribution
            self.map_cilia_to_sphere() # sphere with true distribution
        # else true morph
        
        # Set cilia forces
        if align:
            self.set_aligned_cilia_forces(align_angle, restrict=restricted)
        else: # random
            if restricted:
                self.generate_restricted_cilia_forces(unit=unit)
            else:
                self.set_random_cilia_forces()

    def map_cilia_to_sphere(self):

        # compute COM of true morphology, round to nearest int, and convert to int type
        tm_com = [int(x) for x in np.around(center_of_mass(self.true_morph))]

        n_ciliated_cells = np.sum(self.true_morph==2)
        # print(n_ciliated_cells)

        cilia_vectors = []
        
        for x in range(self.true_morph.shape[0]):
            for y in range(self.true_morph.shape[1]):
                for z in range(self.true_morph.shape[2]):
                    if self.true_morph[x,y,z]==2: # ciliated cell
                        cilia_vectors.append((x-tm_com[0],y-tm_com[1],z-tm_com[2]))
        
        # compute COM of sphere
        sphere_com = [int(x) for x in np.around(center_of_mass(self.body))]

        radius = int(np.round(np.cbrt(np.sum(self.body)*3/(4*np.pi))))

        for vec in cilia_vectors:
            # get unit vector and multiply by correct magnitude
            mag = np.sqrt(vec[0]**2+vec[1]**2+vec[2]**2)
            proj = (vec[0]/mag*radius, vec[1]/mag*radius, vec[2]/mag*radius)
            proj = [int(np.round(x)) for x in proj]
            
            x = sphere_com[0]+proj[0]
            y = sphere_com[1]+proj[1] 
            z = sphere_com[2]+proj[2]

            placed = False

            search_dist = 0
            directions = [up,down,front,back,left,right]

            while not placed:
                for direction in directions:
                    if not placed:
                        nx,ny,nz=direction(x,y,z,d=search_dist)
                        if nx < self.body.shape[0] and ny < self.body.shape[1] and nz < self.body.shape[2]:
                            if self.body[nx,ny,nz]==1:
                                self.body[nx,ny,nz]=2
                                placed = True
                search_dist+=1
            
        # print(np.sum(self.body==2))

    def set_random_cilia_forces(self):
        # Random forces array of size body.shape between [-1,1)
        self.cilia = np.random.random(size=(self.body.shape[0], self.body.shape[1], self.body.shape[2], 3))*2-1
        
        # Only assign forces to ciliated cells (material ID = 2)
        self.cilia[:,:,:,0][self.body!=2] = 0
        self.cilia[:,:,:,1][self.body!=2] = 0
        self.cilia[:,:,:,2] = 0 # no cilia forces in the z direction

        # Make unit vector
        for i in range(self.cilia.shape[0]):
            for j in range(self.cilia.shape[1]):
                for k in range(self.cilia.shape[2]):
                    cell = self.cilia[i,j,k,:]
                    x_comp = cell[0]
                    y_comp = cell[1] 
                    
                    assert cell[2]==0 # assert no z-component of cilia force
                    
                    if x_comp != 0 and y_comp != 0:
                        # Compute unit vector components
                        ux = x_comp/np.linalg.norm([x_comp,y_comp])
                        uy = y_comp/np.linalg.norm([x_comp,y_comp])

                        self.cilia[i,j,k,0] = ux     
                        self.cilia[i,j,k,1] = uy

 
    def set_aligned_cilia_forces(self, align_angle, restrict):
        
        # Zeros array of size body.shape to hold cilia forces
        self.cilia = np.zeros((self.body.shape[0], self.body.shape[1], self.body.shape[2], 3))
        
        # If align_degree_random=0, then cilia forces are as aligned as possible with restriction of being push-only

        # Single random cilia force between (as angle between 0-2pi) to be assigned to all ciliated cells

        if align_angle is not None:
            assert align_angle>=0
            assert align_angle<=2*np.pi
            force = align_angle
        else:
            force = np.random.random()*(np.pi*2)

        # For testing
        # return force
        
        force_neg = np.abs(force-(2*np.pi)) # the other option for how the force could be represented
        # print(force, force_neg)
        # exit()
        
        # print(force)
        # print(force_neg)
        # force_in_degrees = force*(180/np.pi)
        # print(force_in_degrees)

        if restrict:
            rad45 = (45/180)*np.pi
            
            # iterate through ciliated cells
            for r in range(self.body.shape[0]):
                for c in range(self.body.shape[1]):
                    for z in range(self.body.shape[2]):
                        
                        if self.body[r,c,z]==2:

                            # TESTING
                            # print(r,c)
                            # plt.matshow(body[:,:,z], origin='lower')
                            # plt.plot(c,r, '*b') # col is the x val in a plot, row is the y
                            curr_pos = (r,c,z)
                            # Get neighboring empty voxel locations
                            empty_neigh = self.get_empty_neighbor_positons(self.body,curr_pos)

                            # Compute vectors to directions of the empty neighbors
                            vectors = []
                            for empty_neigh_pos in empty_neigh:
                                
                                # TESTING
                                # plt.plot(empty_neigh_pos[1], empty_neigh_pos[0], '*m') # col=x, row=y

                                x_comp = curr_pos[1] - empty_neigh_pos[1]
                                y_comp = curr_pos[0] - empty_neigh_pos[0]
                                z_comp = curr_pos[2] - empty_neigh_pos[2]# should always be 0
                                assert z_comp==0
                                
                                # by default all of the vectors are unit vectors because the distance between voxels is 1
                                vectors.append([x_comp,y_comp,z_comp])

                                # plt.quiver(c,r,x_comp,y_comp,angles=math.degrees(math.atan2(y_comp,x_comp)), scale=5)
                            
                            # plt.show()
                            # exit()
                        
                            # Compute range of angles the cilia force vector can lie in
                            # +/- 45 degrees of the vector to each empty neighboring voxel
                            force_in_bounds = False # keep track of whether the force is in any of the bounds
                            if len(vectors)>0:
                                bounds = []
                                for vector in vectors:

                                    # print("VECTOR:",vector)

                                    angle_in_radians = np.arctan2(vector[1],vector[0])
                                    lb = angle_in_radians-rad45 # lower bound
                                    ub = angle_in_radians+rad45 # upper bound
                                    
                                    bounds.append([lb,ub])

                                    # Check to see if the chosen force for alignment is within the bounds
                                    if force>lb and force<ub:
                                        force_in_bounds=True
                                        cilia_force_angle=force
                                    elif force_neg>lb and force_neg<ub:
                                        force_in_bounds=True
                                        cilia_force_angle=force_neg

                                # print(bounds)
                                # for bound in bounds:
                                #     print(bound[0]*(180/np.pi), bound[1]*(180/np.pi))
                                # print(force_in_bounds)

                                if not force_in_bounds:
                                    # Choose the closest force in the bounds 
                                    dist_from_bounds=100000 # set to something really high

                                    for bound in bounds:

                                        # check both ends of the bound
                                        
                                        dist_from_bounds0 = np.min((np.abs(bound[0]-force), np.abs(bound[0]-force_neg)))                        
                                        dist_from_bounds1 = np.min((np.abs(bound[1]-force), np.abs(bound[1]-force_neg)))

                                        if dist_from_bounds0 == dist_from_bounds1 and dist_from_bounds0 < dist_from_bounds:
                                            closest_bound = np.random.choice([bound[0], bound[1]])
                                            dist_from_bounds = dist_from_bounds0 # because they are the same
                                        else: # if neither of them are closer then don't change anything
                                            if dist_from_bounds0 < dist_from_bounds:
                                                closest_bound=bound[0]
                                                dist_from_bounds = dist_from_bounds0
                                            if dist_from_bounds1 < dist_from_bounds:
                                                closest_bound=bound[1]
                                                dist_from_bounds = dist_from_bounds1
                                    
                                    cilia_force_angle = closest_bound
                                
                                # print(cilia_force_angle)

                                # Compute the x and y components of the unit vector given the chosen angle
                                cilia_x_comp = np.cos(cilia_force_angle) # row is really the y comp 
                                cilia_y_comp = np.sin(cilia_force_angle) # c is really the x comp
                                cilia_z_comp = 0
                                cilia_force_vec = [cilia_x_comp, cilia_y_comp, cilia_z_comp]                      

                                # print("CILIA UNIT VECTOR:",cilia_force_vec)
                                
                                self.cilia[r,c,z,:] = cilia_force_vec

                                # plot_cilia_force_vectors(self.body, self.cilia, l=z)
        else: # unrestricted 
            cilia_x_comp = np.cos(force) # row is really the y comp 
            cilia_y_comp = np.sin(force) # c is really the x comp
            cilia_z_comp = 0
            cilia_force_vec = [cilia_x_comp, cilia_y_comp, cilia_z_comp]                      
            
            self.cilia[self.body==2,:] = cilia_force_vec
        
    def get_surface_cell_coords(self, arr):
        # Get the coordinates of cells on the surface of the bot
        xs = []
        ys = []
        zs = []
        neigh = self.get_neighbors(arr)

        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                for z in range(arr.shape[2]):
                    if arr[x,y,z]==1 and np.sum(neigh[x,y,z,:])<6:
                        xs.append(x)
                        ys.append(y)
                        zs.append(z)

        return xs,ys,zs

    def randomize_cilia_distribution(self):

        temp = self.body.copy()

        # Count total number of cells and number of ciliated cells
        unique, counts = np.unique(self.true_morph, return_counts=True)
        cell_count_dict = dict(zip(unique, counts))
        tot_cells = cell_count_dict[1]+cell_count_dict[2]
        n_ciliated_cells = cell_count_dict[2]

        # Remove all ciliated cells
        temp[temp==2]=1

        # Get the coordinates of cells on the surface of the bot
        xs,ys,zs = self.get_surface_cell_coords(temp)

        dim = len(xs)
        surface_cell_coords = np.concatenate((np.array(xs).reshape((dim,1)), np.array(ys).reshape((dim,1)), np.array(zs).reshape((dim,1))),axis=1)

        # If there aren't enough surface cells, choose from the second layer as well
        if n_ciliated_cells/len(surface_cell_coords) > 0.9:
            mask = temp.copy()
            mask[xs,ys,zs]=0

            n_xs, n_ys, n_zs = self.get_surface_cell_coords(mask)
            dim = len(n_xs)
            second_layer = np.concatenate((np.array(n_xs).reshape((dim,1)), np.array(n_ys).reshape((dim,1)), np.array(n_zs).reshape((dim,1))),axis=1)

            surface_cell_coords = np.concatenate((surface_cell_coords, second_layer))

        # Choose n_ciliated_cells random rows 
        ciliated_cell_rows = np.random.choice(surface_cell_coords.shape[0], size=n_ciliated_cells, replace=False)

        # Set the randomly chosen surface cells to material ID 2 (cilia)
        for r in ciliated_cell_rows:
            temp[surface_cell_coords[r,0], surface_cell_coords[r,1], surface_cell_coords[r,2]]=2

        # Check to make sure total number of cells and number of ciliated cells are preserved and that the body has changed
        unique, counts = np.unique(temp, return_counts=True)
        cell_count_dict_new = dict(zip(unique, counts))
        tot_cells_new = cell_count_dict_new[1]+cell_count_dict_new[2]
        n_ciliated_cells_new = cell_count_dict_new[2]

        assert np.sum(self.body>0) == tot_cells_new
        assert n_ciliated_cells == n_ciliated_cells_new
        assert np.any(temp!=self.body)

        self.body = temp

    def get_neighbors(self, a):
        b = np.pad(a, pad_width=1, mode='constant', constant_values=0)
        neigh = np.concatenate((
            b[2:, 1:-1, 1:-1, None], b[:-2, 1:-1, 1:-1, None],
            b[1:-1, 2:, 1:-1, None], b[1:-1, :-2, 1:-1, None],
            b[1:-1, 1:-1, 2:, None], b[1:-1, 1:-1, :-2, None]), axis=3)
        return neigh

    def get_empty_neighbor_positons(self,body,pos):
        empty_neigh = []
        for direction in [front,back,left,right]:
            neigh_pos = direction(pos[0],pos[1],pos[2])

            # Checking if the neighboring voxels is in array bounds
            if neigh_pos[0]>=0 and neigh_pos[0]<body.shape[0] and neigh_pos[1]>=0 and neigh_pos[1]<body.shape[1]:

                if body[neigh_pos] == 0: # in bounds and empty
                    empty_neigh.append(neigh_pos)

            else: # out of array bounds so by default there is an empty neighbor
                empty_neigh.append(neigh_pos)

        return empty_neigh


    def generate_restricted_cilia_forces(self, set_force=False, align_vec=None, align_angle=None, unit=False):
        # Assumes voxels do not have cilia forces in the z-direction
        # Straight --> cilia force will be at a 90 degree angle from the surface (i.e. just chooses
        # a unit vector to an empty neighbor)
        # if align_vec is not None and straight=True, and a cell has more than one empty neighbor
        # the straight vector is chosen as determined by the align_vec

        self.cilia = np.zeros((self.body.shape[0], self.body.shape[1], self.body.shape[2], 3))

        rad45 = (45/180)*np.pi
        
        # iterate through ciliated cells
        for r in range(self.body.shape[0]):
            for c in range(self.body.shape[1]):
                for z in range(self.body.shape[2]):
                    
                    if self.body[r,c,z]==2:

                        # TESTING
                        # print(r,c)
                        # plt.matshow(body[:,:,z], origin='lower')
                        # plt.plot(c,r, '*b') # col is the x val in a plot, row is the y
                        curr_pos = (r,c,z)
                        # Get neighboring empty voxel locations
                        empty_neigh = self.get_empty_neighbor_positons(self.body,curr_pos)

                        # Compute vectors to directions of the empty neighbors
                        vectors = []
                        for empty_neigh_pos in empty_neigh:
                            
                            # TESTING
                            # plt.plot(empty_neigh_pos[1], empty_neigh_pos[0], '*m') # col=x, row=y

                            x_comp = curr_pos[1] - empty_neigh_pos[1]
                            y_comp = curr_pos[0] - empty_neigh_pos[0]
                            z_comp = curr_pos[2] - empty_neigh_pos[2]# should always be 0
                            assert z_comp==0
                            
                            # by default all of the vectors are unit vectors because the distance between voxels is 1
                            vectors.append([x_comp,y_comp,z_comp])

                            # plt.quiver(c,r,x_comp,y_comp,angles=math.degrees(math.atan2(y_comp,x_comp)), scale=5)
                        
                        # plt.show()
                        # exit()
                        
                        # print(vectors)
                        if set_force:
                            if align_vec is not None:
                                # all vectors are aligned based on the align_vec
                                # print(align_vec)
                                self.cilia[r,c,z,:] = align_vec
                            
                            elif align_angle is not None:
                                cilia_force_vec = [np.cos(align_angle),np.sin(align_angle),0]
                                # print(cilia_force_vec)
                                self.cilia[r,c,z,:] = cilia_force_vec
                            else:
                                # Just choose one of the vectors to the neighboring cells 

                                # Random perpendicular vectors
                                # if len(vectors)>0:
                                #     straight_vec = vectors[np.random.randint(0,len(vectors))]
                                #     assert straight_vec[2] == 0
                                #     self.cilia[r,c,z,:] = straight_vec

                                # DEBUG: perpendicular force (symmetrical bot should not move if cilia is implemented correctly)
                                # how much it moves is a measure of asymmetry that only requires a single evaluation
                                if len(vectors)==1:
                                    self.cilia[r,c,z,:] = vectors[0]
                                elif len(vectors)==2:
                                    self.cilia[r,c,z,:] = [a+b for a,b in zip(vectors[0],vectors[1])]  # sum vectors so cilia points inward to center of body

                        else:
                        # Compute range of angles the cilia force vector can lie in
                        # +/- 45 degrees of the vector to each empty neighboring voxel
                            if len(vectors)>0:
                                bounds = []
                                for vector in vectors:

                                    # print("VECTOR:",vector)

                                    angle_in_radians = np.arctan2(vector[1],vector[0])
                                    lb = angle_in_radians-rad45 # lower bound
                                    ub = angle_in_radians+rad45 # upper bound
                                    
                                    bounds.append([lb,ub])

                                # print(bounds)

                                # first choose a random range to choose from if more than one 
                                # important if the ranges are not touching
                                # i.e. missing voxels to the left and right but not up and down
                                range_index = np.random.randint(len(bounds))
                                angle_range = bounds[range_index]

                                # print(angle_range)

                                # Choose a random angle in the range (on the unit circle)
                                cilia_force_angle = (angle_range[1] - angle_range[0]) * np.random.random() + angle_range[0]
                                
                                # Compute the x and y components of the unit vector given the chosen angle
                                cilia_x_comp = np.cos(cilia_force_angle) # row is really the y comp 
                                cilia_y_comp = np.sin(cilia_force_angle) # c is really the x comp
                                cilia_z_comp = 0
                                cilia_force_vec = [cilia_x_comp, cilia_y_comp, cilia_z_comp]                      

                                # print("CILIA UNIT VECTOR:",cilia_force_vec)
                                
                                if not unit:
                                    cilia_force_vec = [x*50 for x in cilia_force_vec]

                                print(np.linalg.norm(cilia_force_vec))
                                exit()

                                self.cilia[r,c,z,:] = cilia_force_vec

        # return cilia

    def remove_padding(self,arr):
        pad_value = 0

        x=0
        while np.all(arr[x,:,:]==pad_value):
            arr = arr[x+1:,:,:]
            x+=1
        x=arr.shape[0]-1
        while np.all(arr[x,:,:]==pad_value):
            arr = arr[:x,:,:]
            x-=1

        y=0
        while np.all(arr[:,y,:]==pad_value):
            arr = arr[:,y+1:,:]
            y+=1
        y=arr.shape[1]-1
        while np.all(arr[:,y,:]==pad_value):
            arr = arr[:,:y,:]
            y-=1

        z=0
        while np.all(arr[:,:,z]==pad_value):
            arr = arr[:,:,z+1:]
            z+=1
        z=arr.shape[2]-1
        while np.all(arr[:,:,z]==pad_value):
            arr = arr[:,:,:z]
            z-=1

        return arr
