import taichi as ti
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import savemat

from viz import visualize

arch = ti.gpu # Use ti.metal if you are on Apple M1, ti.gpu if using CUDA
# arch = ti.amdgpu
real = ti.f32
dim = 3
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

# TODO: generalize this 
TEST_BOT_PKL_FILE_PATH = './pickle/circular/Run6group5subject1/Run6group5subject1_res30.p'


@ti.func
def isnan(x):
    return not (x < 0 or 0 < x or x == 0)


@ti.func
def zero_vec():
    return [0.0, 0.0, 0.0]


@ti.func
def zero_matrix():
    return [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]


@ti.data_oriented
class DiffControl:
    # n_grid = 96
    dx = 1 / 128
    inv_dx = 1 / dx
    p_vol = 1
    E = 50
    nu = 0.1
    mu = E / (2 * (1 + nu))
    la = E * nu / ((1 + nu) * (1 - 2 * nu))
    max_steps = 2048
    visu_steps = 2048
    steps = 2048
    coeff = 0.5
    bound = 3
    # These will be overwritten by the Robot object
    n_particles = 0
    n_solid_particles = 0
    n_actuators = 0
    is_initialized = False

    # Track losses for all iterations 
    losses = []

    def __init__(self, savedata_folder=None, experiment_parameters=None):
        # Initialize TaiChi 
        ti.init(
            default_fp=real, 
            arch=arch, 
            device_memory_GB=4.5, 
            flatten_if=True
        ) 
        
        # Initialize save folder
        self.folder = savedata_folder
        if savedata_folder is not None:
            os.makedirs(savedata_folder, exist_ok=True)

        # Extract experiment-specific parameters
        self.dt = experiment_parameters['dt']
        self.gravity = experiment_parameters['gravity']
        self.actuation_omega = experiment_parameters['actuation_omega']
        self.act_strength = experiment_parameters['actuation_strength']
        self.learning_rate = experiment_parameters['learning_rate']
        self.actuation_axes = experiment_parameters['actuation_axes']
        self.n_grid = int(experiment_parameters['grid_res'])
        self.n_actuation_axes = len(self.actuation_axes)

        # Initialize memory for TaiChi simulation
        self.actuator_id = ti.field(ti.i32)
        self.particle_type = ti.field(ti.i32)
        self.x, self.v = vec(), vec()
        self.grid_v_in, self.grid_m_in = vec(), scalar()
        self.grid_v_out = vec()
        self.C, self.F = mat(), mat()

        self.loss = scalar()

        # self.n_sin_waves = 4
        self.weights = scalar()
        self.bias = scalar()
        self.offsets = scalar()
        # self.omegas = scalar()
        self.x_avg = vec()

        self.cauchy_act = mat() # tracking cauchy stress matrix over time

        self.actuation_x, self.actuation_y, self.actuation_z = scalar(), scalar(), scalar()
        # self.actuation_value = scalar()

    def allocate_fields(self):
        """
        Allocates fields for TaiChi simulation
        """
        # ti.root.dense(ti.ij, (self.n_actuators, self.n_sin_waves)).place(self.weights)
        ti.root.dense(ti.ij, (self.n_actuation_axes, self.n_actuators)).place(self.bias)
        ti.root.dense(ti.ij, (self.n_actuation_axes, self.n_actuators)).place(self.weights)
        ti.root.dense(ti.ij, (self.n_actuation_axes, self.n_actuators)).place(self.offsets)
        # ti.root.dense(ti.ij, (self.n_actuation_axes, self.n_actuators)).place(self.omegas)

        ti.root.dense(ti.ij, (self.max_steps, self.n_actuators)).place(self.actuation_x)
        # if self.n_actuation_axes > 1:
        ti.root.dense(ti.ij, (self.max_steps, self.n_actuators)).place(self.actuation_y)
        # if self.n_actuation_axes > 2:
        ti.root.dense(ti.ij, (self.max_steps, self.n_actuators)).place(self.actuation_z)
        # ti.root.dense(ti.ijk, (self.n_actuation_axes, self.max_steps, self.n_actuators)).place(self.actuation_value)
        ti.root.dense(ti.i, self.n_particles).place(self.actuator_id, self.particle_type)
        ti.root.dense(ti.k, self.max_steps).dense(ti.l, self.n_particles).place(self.x, self.v, self.C, self.F)
        ti.root.dense(ti.ijk, (self.n_grid, 32, self.n_grid)).place(self.grid_v_in, self.grid_m_in, self.grid_v_out)
        ti.root.place(self.loss, self.x_avg)
        ti.root.dense(ti.ij, (self.max_steps, self.n_actuators)).place(self.cauchy_act)
        ti.root.lazy_grad()

    @ti.kernel
    def clear_grid(self):
        for i, j, k in self.grid_m_in:
            self.grid_v_in[i, j, k] = [0, 0, 0]
            self.grid_m_in[i, j, k] = 0
            self.grid_v_in.grad[i, j, k] = [0, 0, 0]
            self.grid_m_in.grad[i, j, k] = 0
            self.grid_v_out.grad[i, j, k] = [0, 0, 0]

    @ti.kernel
    def clear_particle_grad(self):
        # for all time steps and all particles
        for f, i in self.x:
            self.x.grad[f, i] = zero_vec()
            self.v.grad[f, i] = zero_vec()
            self.C.grad[f, i] = zero_matrix()
            self.F.grad[f, i] = zero_matrix()

    @ti.kernel
    def clear_actuation_grad(self):
        for a, t, i in self.actuation:
            self.actuation[a, t, i] = 0.0

    @ti.kernel
    def p2g(self, f: ti.i32):
        for p in range(self.n_particles):
            base = ti.cast(self.x[f, p] * self.inv_dx - 0.5, ti.i32)
            fx = self.x[f, p] * self.inv_dx - ti.cast(base, ti.i32)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_F = (ti.Matrix.diag(dim=dim, val=1) + self.dt * self.C[f, p]) @ self.F[f, p]
            J = new_F.determinant()
            if self.particle_type[p] == 0:  # fluid
                sqrtJ = ti.sqrt(J)
                # TODO: need pow(x, 1/3)
                new_F = ti.Matrix([[sqrtJ, 0, 0], [0, sqrtJ, 0], [0, 0, 1]])

            self.F[f + 1, p] = new_F
            # r, s = ti.polar_decompose(new_F)

            act_id = self.actuator_id[p]
            A = ti.Matrix.zero(real, 3, 3)

            if act_id > 0:
                act = self.actuation_x[f, ti.max(0, act_id)] * self.act_strength
                A[0, 0] = act
                if self.n_actuation_axes > 1:
                    act = self.actuation_y[f, ti.max(0, act_id)] * self.act_strength
                    A[1, 1] = act
                if self.n_actuation_axes > 2:
                    act = self.actuation_z[f, ti.max(0, act_id)] * self.act_strength
                    A[2, 2] = act
                # if self.n_actuation_axes > 1:
                #     act = self.actuation[0, f, ti.max(0, act_id)] * self.act_strength
                #     A[1, 1] = act
                # for axis in range(self.n_actuation_axes):
                #     act = self.actuation[axis, f, ti.max(0, act_id)] * self.act_strength
                #     # A[dim - 2, dim - 2] = act
                #     A[axis, axis] = act

            cauchy = new_F @ A @ new_F.transpose()
            if self.particle_type != 0:
                self.cauchy_act[f, self.actuator_id[p]] = cauchy

            mass = 1
            if self.particle_type[p] == 0:
                # mass = 4
                # mass = 1
                mass = 4
                cauchy += ti.Matrix.identity(real, dim) * (J - 1) * self.E
            else:
                cauchy += self.mu * (new_F @ new_F.transpose()) + ti.Matrix.identity(real, dim) * (self.la * ti.log(J) - self.mu)
            stress = -(self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * cauchy
            affine = stress + mass * self.C[f, p]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        offset = ti.Vector([i, j, k])
                        dpos = (ti.cast(ti.Vector([i, j, k]), real) - fx) * self.dx
                        weight = w[i][0] * w[j][1] * w[k][2]
                        ti.atomic_add(self.grid_v_in[base + offset], weight * (mass * self.v[f, p] + affine @ dpos))
                        ti.atomic_add(self.grid_m_in[base + offset], weight * mass)

    @ti.kernel
    def grid_op(self):
        for i, j, k in self.grid_m_in:
            inv_m = 1 / (self.grid_m_in[i, j, k] + 1e-10)
            v_out = inv_m * self.grid_v_in[i, j, k]
            v_out[1] -= self.dt * self.gravity

            if i < self.bound and v_out[0] < 0:
                v_out[0] = 0
                v_out[1] = 0
                v_out[2] = 0
            if i > self.n_grid - self.bound and v_out[0] > 0:
                v_out[0] = 0
                v_out[1] = 0
                v_out[2] = 0

            if k < self.bound and v_out[2] < 0:
                v_out[0] = 0
                v_out[1] = 0
                v_out[2] = 0
            if k > self.n_grid - self.bound and v_out[2] > 0:
                v_out[0] = 0
                v_out[1] = 0
                v_out[2] = 0

            if j < self.bound and v_out[1] < 0:
                v_out[0] = 0
                v_out[1] = 0
                v_out[2] = 0
                normal = ti.Vector([0.0, 1.0, 0.0])
                lsq = (normal ** 2).sum()
                if lsq > 0.5:
                    if ti.static(self.coeff < 0):
                        v_out[0] = 0
                        v_out[1] = 0
                        v_out[2] = 0
                    else:
                        lin = v_out.dot(normal)
                        if lin < 0:
                            vit = v_out - lin * normal
                            lit = vit.norm() + 1e-10
                            if lit + self.coeff * lin <= 0:
                                v_out[0] = 0
                                v_out[1] = 0
                                v_out[2] = 0
                            else:
                                v_out = (1 + self.coeff * lin / lit) * vit
            if j > self.n_grid - self.bound and v_out[1] > 0:
                v_out[0] = 0
                v_out[1] = 0
                v_out[2] = 0

            self.grid_v_out[i, j, k] = v_out

    @ti.kernel
    def g2p(self, f: ti.i32):
        for p in range(self.n_particles):
            base = ti.cast(self.x[f, p] * self.inv_dx - 0.5, ti.i32)
            fx = self.x[f, p] * self.inv_dx - ti.cast(base, real)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector(zero_vec())
            new_C = ti.Matrix(zero_matrix())

            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        dpos = ti.cast(ti.Vector([i, j, k]), real) - fx
                        g_v = self.grid_v_out[base[0] + i, base[1] + j, base[2] + k]
                        weight = w[i][0] * w[j][1] * w[k][2]
                        new_v += weight * g_v
                        new_C += 4 * weight * g_v.outer_product(dpos) * self.inv_dx

            self.v[f + 1, p] = new_v
            self.x[f + 1, p] = self.x[f, p] + self.dt * self.v[f + 1, p]
            self.C[f + 1, p] = new_C

    @ti.kernel
    def compute_actuation(self, t: ti.i32):
        for i, axis in ti.ndrange(self.n_actuators, self.n_actuation_axes):
            act = self.weights[axis, i] * ti.sin(self.actuation_omega * t * self.dt + self.offsets[axis, i])
            # for j in ti.static(range(self.n_sin_waves)):
            #     act += self.weights[i, j] * ti.sin(self.actuation_omega * t * self.dt + 2 * math.pi / self.n_sin_waves * j)
            act += self.bias[axis, i]

            # Track the actuation of a single particle
            # self.actuation_value[axis, t, i] = act

            # Activated actuation
            if axis == 0:
                self.actuation_x[t, i] = ti.tanh(act)
            if axis == 1:
                self.actuation_y[t, i] = ti.tanh(act)
            if axis == 2:
                self.actuation_z[t, i] = ti.tanh(act)

    @ti.kernel
    def compute_x_avg(self, steps: ti.i32):
        for i in range(self.n_particles):
            contrib = 0.0
            if self.particle_type[i] == 1:
                contrib = 1.0 / self.n_solid_particles
            ti.atomic_add(self.x_avg[None], contrib * self.x[steps - 1, i])

    @ti.kernel
    def compute_loss(self):
        dist = self.x_avg[None][0] ** 2 + self.x_avg[None][2] ** 2
        self.loss[None] = -dist

    def forward(self, total_steps=steps):
        # simulation
        for t in range(total_steps - 1):
            self.clear_grid()
            self.compute_actuation(t)
            self.p2g(t)
            self.grid_op()
            self.g2p(t)

        self.x_avg[None] = [0, 0, 0]
        self.compute_x_avg(self.steps)
        self.compute_loss()
        return self.loss[None]

    def backward(self):
        self.clear_particle_grad()

        self.compute_loss.grad()
        self.compute_x_avg.grad(self.steps)
        for s in reversed(range(self.steps - 1)):
            # Since we do not store the grid history (to save space), we redo p2g and grid op
            self.clear_grid()
            self.p2g(s)
            self.grid_op()

            self.g2p.grad(s)
            self.grid_op.grad()
            self.p2g.grad(s)
            self.compute_actuation.grad(s)

    @ti.kernel
    def learn(self, learning_rate: ti.template()):
        # for i, j in ti.ndrange(self.n_actuators, self.n_sin_waves):
        #     self.weights[i, j] -= learning_rate * self.weights.grad[i, j]
        #
        # for i in range(self.n_actuators):
        #     self.bias[i] -= learning_rate * self.bias.grad[i]
        for i in range(self.n_actuators):
            for axis in range(self.n_actuation_axes):
                self.bias[axis,i] -= learning_rate * self.bias.grad[axis,i]
                self.weights[axis,i] -= learning_rate * self.weights.grad[axis,i]
                self.offsets[axis,i] -= learning_rate * self.offsets.grad[axis,i]
                # self.omegas[axis,i] -= learning_rate * self.omegas.grad[axis,i]

    def init(self, robot, load_weights_filename=None, savedata_folder=None):
        self.n_particles = robot.n_particles
        self.n_actuators = robot.n_actuators
        self.n_solid_particles = robot.n_solid_particles
        print('n_particles', self.n_particles)
        print('n_solid', self.n_solid_particles)
        print('n_actuators', self.n_actuators)
        print('actuator proportion: ', self.n_actuators / self.n_particles)
        self.allocate_fields()

        # for i, j in ti.ndrange(self.n_actuators, self.n_sin_waves):
        #     self.weights[i, j] = np.random.randn() * 0.01
        for i, axis in ti.ndrange(self.n_actuators, self.n_actuation_axes):
            # for axis in range(self.n_actuation_axes):
            self.weights[axis, i] = np.random.randn() * 0.1
            self.offsets[axis, i] = np.random.randn()
            # self.omegas[axis, i] = np.random.randn() * 100
            # print(self.weights[i], self.offsets[i], self.omegas[i])

        for i in range(self.n_particles):
            self.x[0, i] = robot.x[i]
            self.F[0, i] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            self.actuator_id[i] = robot.actuator_id[i]
            self.particle_type[i] = robot.particle_type[i]

        self.prepare_exp(load_weights_filename, savedata_folder)
        self.is_initialized = True

    def prepare_exp(self, load_weights_filename=None, savedata_folder=None):
        if savedata_folder is not None:
            self.folder = savedata_folder
            os.makedirs(savedata_folder, exist_ok=True)

        if load_weights_filename is not None:
            with open(f'{load_weights_filename}', 'rb') as f:
                self.weights.from_numpy(np.load(f))
                self.bias.from_numpy(np.load(f))

    def save_weights(self, iters, loss_val):
        with open(f'{self.folder}/iter{iters:04d}_{loss_val:04d}.npy', 'wb') as f:
            np.save(f, self.weights.to_numpy())
            np.save(f, self.bias.to_numpy())

    def run(self, iters, visualize=False):
        for it in range(iters):
            t = time.time()
            ti.ad.clear_all_gradients()
            loss = self.forward(self.steps)
            self.losses.append(loss)
            self.loss.grad[None] = 1
            self.backward()
            per_iter_time = time.time() - t
            
            print('i=', it, 'loss=', loss, F' per iter {per_iter_time:.2f}s')

            if np.isnan(loss):
                print('Loss is NaN. Exiting...')
                break

            self.learn(self.learning_rate)

            if visualize and it % 50 == 0:
                print('Writing particle data to disk...')
                print('(Please be patient)...')
                x_numpy = self.x.to_numpy()
                visualize(self.actuator_id, x_numpy, it)
                # ti.profiler.print_kernel_profiler_info()

        plt.title("Optimization of Initial Velocity")
        plt.ylabel("Loss")
        plt.xlabel("Gradient Descent Iterations")
        plt.plot(self.losses)
        plt.savefig(f'{self.folder}/loss_fig.png')

    def run_once(self):
        assert self.is_initialized
        self.forward(self.visu_steps)
        loss_val = self.x_avg[None][0]
        if self.folder is not None:
            # savemat(f'{self.folder}.mat', {'x': self.x.to_numpy(), 'is_coll': self.is_coll.to_numpy()})
            savemat(f'{self.folder}.mat', {'x': np.squeeze(np.mean(self.x.to_numpy()[:self.visu_steps, :, :], axis=1))})
        print("loss=", loss_val)
        return loss_val

    def pickle_positions(self, file_name):
        position_data = self.x.to_numpy()
        actuator_ids = self.actuator_id.to_numpy()
        data_obj = (position_data, actuator_ids)
        with open(f'{self.folder}/{file_name}', 'wb') as pf:
            pickle.dump(data_obj, pf, protocol=pickle.HIGHEST_PROTOCOL)

    def pickle_act(self, file_name):
        act_data = self.cauchy_act.to_numpy()
        actuator_ids = self.actuator_id.to_numpy()
        data_obj = (act_data, actuator_ids)
        with open(f'{self.folder}/{file_name}', 'wb') as pf:
            pickle.dump(data_obj, pf, protocol=pickle.HIGHEST_PROTOCOL)

    def pickle_loss(self, file_name):
        with open(f'{self.folder}/{file_name}', 'wb') as pf:
            pickle.dump(self.losses, pf, protocol=pickle.HIGHEST_PROTOCOL)

        