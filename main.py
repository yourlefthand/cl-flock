import sys, cProfile

import pyopencl as cl

import numpy as np
import scipy.misc as scp

def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func

def init_data(num, res_x, res_y):
    pos = np.zeros((num, 4), dtype=np.int32)

    #pos[:,0] = np.arange(num) % np.sqrt(num)
    pos[:,0] = np.arange(num) % res_x
    pos[:,0] *= np.random.random_sample((num,))

    pos[:,1] = np.arange(num) % res_y
    pos[:,1] *= np.random.random_sample((num,))

    pos[:,2] = 0
    pos[:,3] = 0

    mass = np.zeros(num, dtype=np.int32)
    mass[:] = 12

    power = np.zeros(num, dtype=np.int32)
    power[:] = 36

    lifetime = np.zeros(num, dtype=np.int32)
    lifetime[:] = 10

    return pos, mass, power, lifetime

class KDTree(object):
    """requires dataset with length of rational sqrt for now"""
    def __init__(self, min_leaf_size=2):
        self.min_leaf_size = min_leaf_size
    def recurse(self, data, shape, dimension):
        if shape[0] % self.min_leaf_size == 0 and shape[1] % self.min_leaf_size == 0:
            med = np.divide(shape[dimension],2)
            left = np.where((data[:,dimension] <= med))
            self.tree.append(left[:])
            right = np.where((data[:,dimension] >  med))
            self.tree.append(right[:])
            left_data  = data[left]
            self.literal.append(left_data[:])
            right_data = data[right]
            self.literal.append(right_data[:])
            shape = shape[:]
            shape[dimension] = med
            dimension += 1
            if dimension > 1:
                dimension = 0
            self.recurse(left_data, shape, dimension)
            self.recurse(right_data, shape, dimension)

    def execute(self, data, shape):
        self.tree = []
        self.literal = []
        dimension = 0

        self.recurse(data, shape, dimension)

        return self.tree, self.literal


class OpenCl(object):
    def __init__(self, num, dt):
        self.num = num
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

    def cl_load_data(self, pos, mass, power, lifetime, world):
        mf = cl.mem_flags

        world_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=world.flatten())

        pos_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pos)
        mass_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mass)
        power_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=power)
        lifetime_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lifetime)

        out = cl.Buffer(self.ctx, mf.WRITE_ONLY, pos.nbytes)

        return pos_cl, mass_cl, power_cl, lifetime_cl, world_cl, out

    def execute(self, pos, mass, power, lifetime, world):
        pos_cl, mass_cl, power_cl, lifetime_cl, world_cl, out = self.cl_load_data(pos, mass, power, lifetime, world)

        inner_rad = np.int32(1)
        outer_rad = np.int32(3)

        world_x = np.int32(world.shape[0])
        world_y = np.int32(world.shape[1])

        ret = np.zeros_like(pos)

        global_size = (self.num,)
        local_size = None

        kernalargs = (pos_cl,
                      mass_cl,
                      power_cl,
                      lifetime_cl,
                      world_cl,
                      out,
                      inner_rad, outer_rad,
                      world_x, world_y
                      )

        self.program.flock(self.queue, global_size, local_size, *(kernalargs)).wait()
        cl.enqueue_read_buffer(self.queue, out, ret)
        #finish not needed with wait
        #self.queue.finish()
        return ret

def form_world(accel_vector, x_size, y_size):
    world = np.zeros((x_size, y_size), dtype=np.int32)
    world[accel_vector[:,0],accel_vector[:,1]] = 1
    return world

def draw(image,count):
    scp.imsave("./out/image/frame_{0:05d}.png".format(count),image.astype(bool))
    #dimen = np.sqrt(num)
    #world = np.zeros((dimen,dimen))
    #for starling in pos:
    #    if starling[0] < dimen and starling[0] > 0 and starling[1] < dimen and starling[1] > 0:
    #        world[starling[0],starling[1]] = True
    #scp.imsave("./out/image/frame_{0:05d}.png".format(count),world.astype(bool))


if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan,linewidth=1024)

    sys.settrace
    sys.setrecursionlimit(1024**2)
    
    num = 1024**2
    resolution = [1024,1024]

    #num = 1280 * 720
    #resolution = [1280,720]
    
    #num = 64 * 64
    #resolution = [64,64]

    #num = 640 * 480
    #resolution = [640, 480]

    #num = 200*200
    #resolution = [200,200]

    print num
    print resolution

    dt = .001

    opcl = OpenCl(num, dt)

    pos, mass, power, lifetime = init_data(num,resolution[0],resolution[1])

    world = form_world(pos,resolution[0],resolution[1])

    #kd = KDTree(min_leaf_size=4)
    #kd.execute(pos, resolution)   

    count = 0
    
    while True:
        #draw(world, count)

        result = opcl.execute(pos, mass, power, lifetime, world)
        print result
        #world = form_world(result,resolution[0],resolution[1])

        #print world
        print count
        #kd.execute(ret, resolution)
        pos = result 
        count += 1
