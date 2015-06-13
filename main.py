import sys

import pyopencl as cl

import numpy as np
import scipy.misc as scp

def init_data(num, res_x, res_y):
    pos = np.zeros((num, 8), dtype=np.int32)

    pos[:,0] = np.arange(num) % res_x
    pos[:,0] *= np.random.random_sample((num,))

    pos[:,1] = np.arange(num) % res_y
    pos[:,1] *= np.random.random_sample((num,))

    pos[:,2] = 3
    pos[:,3] = 9

    return pos

def draw(world, count):
    scp.imsave("./out/image/frame_{0:05d}.png".format(count),world.astype(bool))

def form_world(accel_vector, x_size, y_size):
    world = np.zeros((x_size, y_size), dtype=np.int32)
    world[accel_vector[:,0],accel_vector[:,1]] = 1
    return world

class KDTree(object):
    """requires dataset with length of rational sqrt for now"""
    def __init__(self):
        self.null = None

    def form_tree(self, data, mins=None, maxs=None):
        self.data = np.asarray(data)

        assert self.data.shape[1] == 2

        if mins is None:
            mins = data.min(0)
        if maxs is None:
            maxs = data.max(0)

        self.mins = np.asarray(mins)
        self.maxs = np.asarray(maxs)
        self.size_s = self.maxs - self.mins

        self.child1 = None
        self.child2 = None

        if len(data) > 1:
            # sort on the dimension with the largest spread
            largest_dim = np.argmax(self.size_s)
            i_sort = np.argsort(self.data[:, largest_dim])
            new_data = self.data[i_sort, :]

            # find split point
            N = new_data.shape[0]
            split_point = 0.5 * (new_data[N / 2, largest_dim]
                                 + new_data[N / 2 - 1, largest_dim])

            # create subnodes
            mins1 = self.mins.copy()
            mins1[largest_dim] = split_point
            maxs2 = self.maxs.copy()
            maxs2[largest_dim] = split_point

            # Recursively build a KD-tree on each sub-node
            self.child1 = self.form_tree(new_data[N / 2:], mins=mins1, maxs=self.maxs)      
            self.child2 = self.form_tree(new_data[:N / 2], mins=self.mins, maxs=maxs2)

        return np.asarray([data, self.child1, self.child2])

    def tree(self, data):
        tree = []

        np.append(tree, self.form_tree(data))

        return tree


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

    def cl_load_data(self, pos, world):
        mf = cl.mem_flags

        world_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=world.flatten());

        pos_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pos)

        out = cl.Buffer(self.ctx, mf.WRITE_ONLY, pos.nbytes)

        return pos_cl, world_cl, out

    def execute(self, pos, world):
        pos_cl, world_cl, out = self.cl_load_data(pos, world)

        world_x = np.int32(world.shape[0])
        world_y = np.int32(world.shape[1])

        ret = np.zeros_like(pos)

        global_size = (self.num,)
        local_size = None

        kernalargs = (
                      pos_cl,
                      world_cl,
                      out,
                      world_x, world_y
                     )

        self.program.knn(self.queue, global_size, local_size, *(kernalargs)).wait()
        cl.enqueue_read_buffer(self.queue, out, ret)
        #self.queue.finish()
        return ret

if __name__ == "__main__":
    sys.settrace
    sys.setrecursionlimit(1024**2)

    #num = 1024**2
    #resolution = [1024,1024]

    num = 1280 * 720
    resolution = [720, 1280]

    dt = .001

    starlings = init_data(num, resolution[0], resolution[1])
    world = form_world(starlings, resolution[0],resolution[1])

    opcl = OpenCl(num, dt)

    count = 0
    
    while True:
         draw(world, count)
         starlings = opcl.execute(starlings, world)
         print starlings[:,:2]
         
         world = form_world(starlings, resolution[0],resolution[1])        
         # print world
         
         print count
  
         count += 1