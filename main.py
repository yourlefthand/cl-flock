import sys, time

import pyopencl as cl

import tables

import numpy as np
import scipy.misc as scp

def init_data(population_size, resolution):
    population = np.zeros((population_size, 16), dtype=np.int32)

    max_power = 4
    max_mass = 12

    res_x = resolution[0]
    res_y = resolution[1]
    res_z = resolution[2]

    #initial position
    population[:,0] = np.arange(num) % res_x
    population[:,0] *= np.random.random_sample((num,))

    population[:,1] = np.arange(num) % res_y
    population[:,1] *= np.random.random_sample((num,))

    population[:,2] = np.arange(num) % res_z
    population[:,2] *= np.random.random_sample((num,))

    #velocity + position provides vector - i.e. initial debt
    population[:,3] = 0
    population[:,4] = 0
    population[:,5] = 0

    #mass, power randomized? should be part of genome - eventually?
    population[:,6] = np.arange(num) % (max_mass - 1)
    population[:,6] *= np.random.random_sample((num,))
    population[:,6] += 1

    population[:,7] = np.arange(num) % (max_power - 1)
    population[:,7] *= np.random.random_sample((num,))
    population[:,7] += 1

    #genomic weights to be used as bytestrings
    population[:,8:16] = 0 

    return population

def draw(population, resolution, count):
    res_x = resolution[0]
    res_y = resolution[1]

    flat_world = np.zeros((res_x, res_y), dtype=bool)
    flat_world[population[:,0],population[:,1]] = 1
    scp.imsave("./out/image/frame_{0:05d}.png".format(count),flat_world.astype(bool))

def form_world(population, resolution):
    res_x = resolution[0]
    res_y = resolution[1]
    res_z = resolution[2]

    world = np.zeros((res_x, res_y, res_z), dtype=np.int32)
    world[population[:,0],population[:,1], population[:,2]] = 1
    return world

def save_as_hd5(population, resolution):
    h5_out = tables.open_file('./starlings.h5', mode='w', title="Starlings")
    root = ht_out.root
    h5_out.create_array(root, "population", population)
    h5_out.create_array(root, "resolution", resolution)
    h5_out.close()
    
def read_from_hd5():
    h5_in = tables.open_file('./starlings.h5', mode='r')
    population = h5_in.get_node("population").read()
    resolution = h5_in.get_node("resolution").read()
    return population, resolution

class OpenCl(object):
    def __init__(self, dt):
        self.dt = dt

        self.ctx, self.queue = self.cl_init()
        self.program = self.cl_load_program("./kernal.cl")

    def cl_init(self):
        platforms = cl.get_platforms()
        
        ctx = cl.create_some_context()
        
        queue = cl.CommandQueue(ctx)

        return ctx, queue

    def cl_load_program(self, filepath):
        f = open(filepath, 'r')
        fstr = "".join(f.readlines())
        program = cl.Program(self.ctx, fstr).build()
        return program

    def cl_load_data(self, population, world):
        mf = cl.mem_flags


        out = cl.Buffer(self.ctx, mf.WRITE_ONLY, population.nbytes)

        population_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=population)

#        world_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=world.flatten())
        world_cl = cl.image_from_array(self.ctx, world, mode="r")
        
        return population_cl, world_cl, out

    def execute(self, num, population, world):
        population_cl, world_cl, out = self.cl_load_data(population, world)

        world_x = np.int32(world.shape[0])
        world_y = np.int32(world.shape[1])
        world_z = np.int32(world.shape[2])

        inner_rad = np.int32(32)
        outer_rad = np.int32(36)

        constants = np.asarray([0, 0], dtype=np.int32)

        ret = np.zeros_like(population)        

        global_size = ((num),)
        local_size = None

        kernalargs = (
                      population_cl,
                      out,
                      world_cl,
                      world_x, world_y, world_z,
                      inner_rad, outer_rad
                     )

        image_sequence = []

        self.program.knn(self.queue, global_size, local_size, *(kernalargs)) .wait()
        cl.enqueue_read_buffer(self.queue, out, ret)


        #supersitious drivel!
        out.release()
        population_cl.release()
        world_cl.release()
        
        #self.queue.finish()
        return ret

if __name__ == "__main__":
    sys.settrace
    sys.setrecursionlimit(1024**2)

    #np.set_printoptions(threshold=np.nan, linewidth=512)

    #num = 1920 * 1200
    #resolution = [1200, 1920, 64]

    # num = 1024**2
    # resolution = [1024,1024]

    # num = 1280 * 720
    # resolution = [720, 1280]

    #num = 640 * 481
    #resolution = [480, 640]

    # num = 64 * 64
    # resolution = [64, 64]

    #num = 16
    #resolution = [16, 16, 1]

    num = 640 * 480
    resolution = [480, 640, 1]

    # num = 64 * 64
    # resolution = [64, 64, 1]
    #resolution = [64, 64, 9]

    #num = 32
    #resolution = [32, 32, 32]

    #num = 6 * 6 * 1
    #resolution = [6, 6, 1]

    dt = .001

    starlings = init_data(num, resolution)
    world = form_world(starlings, resolution)


    opcl = OpenCl(dt)

    count = 0
   

    while True:
      draw(starlings, resolution, count)
      # print "init"
      # print starlings[:,0:6]
      # starlings = opcl.execute(starlings, world)
      try:
        starlings = opcl.execute(num, starlings, world)
      except cl.RuntimeError, e:
        print str(e)
        np.set_printoptions(threshold=np.nan, linewidth=512)
        print starlings[:,0:15]
        save_as_hd5(starlings, resolution)
        raise
      #starlings = init_data(num, resolution[0], resolution[1])
      # print "res"
      # print "position/velocity"
      #print starlings[:,0:15]
      # print "cohesion"
      # print starlings[:,6:9]
      # print "separation"
      # print starlings[:,9:12]
      # print "observed"
      # print starlings[:,12]
      # print "coheded"
      # print starlings[:,13]
      # print "separated"
      # print starlings[:,14]
      # print world 
      world = form_world(starlings, resolution)

      print "frame: " + str(count)

      count += 1
