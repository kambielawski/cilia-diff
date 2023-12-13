import random
from enum import Enum

from bot import Bot

TEST_BOT_PKL_FILE_PATH = './pickle/circular/Run6group5subject1/Run6group5subject1_res30.p'

class RobotType(Enum):
    ANTH = 1
    ANTH_SPHERE = 2
    CILIA_BALL = 3
    SOLID = 4
    FLUIDISH = 5


class Robot:
    # The definition of the environment
    def __init__(self, robot_type: RobotType):
        # self.body = None
        self.n_particles = 0        # Total number of particles
        self.n_solid_particles = 0  # Number of solid (non-actuated) particles
        self.n_actuators = 0        # Number of actuated particles
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0
        self.offset_z = 0

        if robot_type == RobotType.ANTH or robot_type == RobotType.ANTH_SPHERE:
            self.set_offset(0.05, 0.02, 0)
            bot = Bot(TEST_BOT_PKL_FILE_PATH, make_sphere=robot_type == RobotType.ANTH_SPHERE, make_surface_ciliated=True)
            self.add_anthrbody(bot.remove_padding(bot.body))
        if robot_type == RobotType.CILIA_BALL:
            self.set_offset(0.05, 0.02, 0)
            self.add_body_cilia()
        elif robot_type == RobotType.SOLID:
            self.set_offset(0.05, 0.02, 0)
            self.add_body()
        elif robot_type == RobotType.FLUIDISH:
            self.add_fluidish_body()

    def new_actuator(self):
        self.n_actuators += 1
        return self.n_actuators - 1

    def add_body(self):
        w, h, d = 15, 15, 15
        max_dim = max([w, h, d])
        print(w, h, d)

        sim_body_size = 0.05
        sim_particle_size = sim_body_size / max_dim
        print(sim_particle_size)
        ptype = 1  # all particles are solid

        for x in range(w):
            for y in range(h):
                for z in range(d):
                    if (x - 7) ** 2 + (y - 7) ** 2 + (z - 7) ** 2 < 7 ** 2:
                        self.x.append([
                            self.offset_x + (x + 0.5) * sim_particle_size,
                            self.offset_y + (y + 0.5) * sim_particle_size,
                            self.offset_z + (z + 0.5) * sim_particle_size
                        ])
                        self.particle_type.append(ptype)
                        self.n_particles += 1
                        self.n_solid_particles += int(ptype == 1)
                        if (x - 7) ** 2 + (y - 7) ** 2 + (z - 7) ** 2 < 6 ** 2:  # ciliated/actuator particle
                            self.actuator_id.append(self.new_actuator())
                        else:  # body particle
                            self.actuator_id.append(-1)

    def add_body_cilia(self):
        w, h, d = 15, 15, 15
        max_dim = max([w, h, d])
        print(w, h, d)

        sim_body_size = 0.05
        sim_particle_size = sim_body_size / max_dim
        print(sim_particle_size)
        ptype = 1  # all particles are solid

        for x in range(w):
            for y in range(h):
                for z in range(d):
                    if (x - 7) ** 2 + (y - 7) ** 2 + (z - 7) ** 2 < 7 ** 2:
                        self.x.append([
                            self.offset_x + (x + 0.5) * sim_particle_size,
                            self.offset_y + (y + 0.5) * sim_particle_size,
                            self.offset_z + (z + 0.5) * sim_particle_size
                        ])
                        self.particle_type.append(ptype)
                        self.n_particles += 1
                        self.n_solid_particles += int(ptype == 1)
                        if (x - 7) ** 2 + (y - 7) ** 2 + (z - 7) ** 2 >= 6 ** 2:  # ciliated/actuator particle
                            self.actuator_id.append(self.new_actuator())
                        else:  # body particle
                            self.actuator_id.append(-1)

    def add_anthrbody(self, body):
        # body is n*n*n array
        # 0 is nothing
        # 1 is body particle
        # 2 is actuator particle

        w, h, d = body.shape
        max_dim = max([w, h, d])
        print(w, h, d)

        sim_body_size = 0.1
        sim_particle_size = sim_body_size / max_dim

        ptype = 1  # all particles are solid

        for x in range(w):
            for y in range(h):
                for z in range(d):
                    if body[x, y, z] > 0:
                        self.x.append([
                            self.offset_z + (z + 0.5) * sim_particle_size,
                            self.offset_y + (y + 0.5) * sim_particle_size,
                            self.offset_x + (x + 0.5) * sim_particle_size
                        ])
                        self.particle_type.append(ptype)
                        self.n_particles += 1
                        self.n_solid_particles += int(ptype == 1)
                        if body[x, y, z] == 1:  # body particle
                            self.actuator_id.append(-1)
                        elif body[x, y, z] == 2:  # ciliated/actuator particle
                            self.actuator_id.append(self.new_actuator())
        print(self.n_actuators)

    def add_fluidish_body(self):
        w, h, d = 15, 15, 15
        max_dim = max([w, h, d])
        print(w, h, d)

        sim_body_size = 0.1
        sim_particle_size = sim_body_size / max_dim
        print(sim_particle_size)

        for x in range(w):
            for y in range(h):
                for z in range(d):
                    if (x - 7) ** 2 + (y - 7) ** 2 + (z - 7) ** 2 < 7 ** 2:
                        ptype = 1  # all particles are solid
                        self.x.append([
                            self.offset_x + (x + 0.5) * sim_particle_size,
                            self.offset_y + (y + 0.5) * sim_particle_size,
                            self.offset_z + (z + 0.5) * sim_particle_size
                        ])
                        if (x - 7) ** 2 + (y - 7) ** 2 + (z - 7) ** 2 < 6 ** 2:
                            if random.randint(0, 2) < 1:
                                self.actuator_id.append(self.new_actuator())
                            else:
                                self.actuator_id.append(-1)
                                ptype = 0
                        else:
                            self.actuator_id.append(-1)
                        self.particle_type.append(ptype)
                        self.n_particles += 1
                        self.n_solid_particles += int(ptype == 1)

    def set_offset(self, x, y, z):
        self.offset_x = x
        self.offset_y = y
        self.offset_z = z