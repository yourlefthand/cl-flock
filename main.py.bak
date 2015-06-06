import sys

import pyopencl as cl

import numpy as np
import scipy.misc as scp

def init_data(num):
    pos = np.zeros((num, 4), dtype=np.int32)

    #pos[:,0] = np.arange(num) % np.sqrt(num)
    pos[:,0] = np.arange(num) % 1280
    pos[:,0] *= np.random.random_sample((num,))

    pos[:,1] = np.arange(num) % 720
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

def draw(num, pos, count):
    dimen = np.sqrt(num)
    world = np.zeros((dimen,dimen))
    #for starling in pos:
    #    if starling[0] < dimen and starling[0] > 0 and starling[1] < dimen and starling[1] > 0:
    #        world[starling[0],starling[1]] = True
    #scp.imsave("./out/image/frame_{0:05d}.png".format(count),world.astype(bool))

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

    def cl_load_data(self, pos, mass, power, lifetime):
        mf = cl.mem_flags

        pos_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pos)
        mass_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mass)
        power_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=power)
        lifetime_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lifetime)

        out = cl.Buffer(self.ctx, mf.WRITE_ONLY, pos.nbytes)

        return pos_cl, mass_cl, power_cl, lifetime_cl, out

    def execute(self, pos, mass, power, lifetime):
        pos_cl, mass_cl, power_cl, lifetime_cl, out = self.cl_load_data(pos, mass, power, lifetime)

        inner_rad = np.int32(1)
        outer_rad = np.int32(3)

        ret = np.zeros_like(pos)

        global_size = (self.num,)
        local_size = None

        kernalargs = (pos_cl,
                      mass_cl,
                      power_cl,
                      lifetime_cl,
                      out,
                      inner_rad, outer_rad
                      )

        self.program.knn(self.queue, global_size, local_size, *(kernalargs)).wait()
        cl.enqueue_read_buffer(self.queue, out, ret)
        #self.queue.finish()
        return ret

if __name__ == "__main__":
    sys.settrace
    sys.setrecursionlimit(1024**2)

    #num = 1024**2
    num = 921600
    resolution = [1280,720]
    dt = .001

    pos, mass, power, lifetime = init_data(num)

    kd = KDTree(min_leaf_size=4)
    kd.execute(pos, resolution)

    print kd.tree

    opcl = OpenCl(num, dt)

    count = 0
    
    while True:
         ret = opcl.execute(pos, mass, power, lifetime)
         print count
         print ret
         draw(num, ret, count)
         #kd.execute(ret, resolution)
         pos = ret;
         count += 1