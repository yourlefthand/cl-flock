import sys

import pyopencl as cl

import numpy as np
import scipy.misc as scp

def init_data(num):
    pos = np.zeros((num, 4), dtype=np.int32)

    pos[:,0] = np.arange(num) % np.sqrt(num)
    pos[:,0] *= np.random.random_sample((num,))

    pos[:,1] = np.arange(num) % np.sqrt(num)
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
    def __init__(self, dataset):
        self.raw_data = dataset

    def recurse(self, data, shape, dimension):
        med = np.divide(shape[dimension],2)
        print dimension
        left = np.where((data[:,dimension] >  med))
        right = np.where((data[:,dimension] <= med))
        if shape[0] % 2 == 0 and shape[1] % 2 == 0:
                    shape[dimension] = med
                    dimension += 1
                    if dimension > 1:
                        dimension = 0
            recurse(left, shape, dimension)
            recurse(right, shape, dimension)
        return bin, shape, dimension

    def execute(self):
        self.tree = []
        self.literal = []
        shape = np.asarray([np.sqrt(len(self.raw_data)).astype(int),np.sqrt(len(self.raw_data)).astype(int)])
        dim = 0


        print self.form_up(self.raw_data, shape, dim)

        # while shape[1] % 2 <= 0:
        #     med = np.divide(shape[dim],2)
        #     bin_up, shape, dimension = self.form_up(bin_up, shape, dimension=dim)
        #     tree.append(bin_up)
        #     bin_up = self.raw_data[bin_up]
        #     literal.append(bin_up)
        #     bin_down, shape, dimension = self.form_down(bin_down, shape, dimension=dim)
        #     tree.append(bin_down)
        #     bin_down = self.raw_data[bin_down]
        #     literal.append(bin_down)
        #     bin_up, shape, dimension = self.form_up(bin_down, shape, dimension=dim)
        #     tree.append(bin_up)
        #     bin_up = self.raw_data[bin_up]
        #     literal.append(bin_up)
        #     print bin_up.shape, bin_down.shape, shape, dimension
        #     dimension += 1
        #     if dimension > 1:
        #        dimension = 0
        #     shape[dimension] = med
        #     dim = dimension


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

    num = 16384
    #num = 1024**2
    dt = .001

    pos, mass, power, lifetime = init_data(num)

    kd_tree = KDTree(pos)
    kd_tree.execute()

    opcl = OpenCl(num, dt)

    count = 0
    
    # while True:
    #     ret = opcl.execute(pos, mass, power, lifetime)
    #     print count
    #     print ret
    #     draw(num, ret, count)
    #     pos = ret;
    #     count += 1