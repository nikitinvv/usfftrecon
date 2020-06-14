"""Module for tomography."""

import cupy as cp
import numpy as np
from usfftrecon.radonusfft import radonusfft
import threading
import concurrent.futures as cf
from functools import partial


class SolverTomo(radonusfft):
    """Base class for tomography solvers using the USFFT method on GPU.
    This class is a context manager which provides the basic operators required
    to implement a tomography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attributes
    ----------
    ntheta : int
        The number of projections.    
    n, nz : int
        The pixel width and height of the projection.
    pnz : int
        The number of slice in partitions to process together
        simultaneously.
    ngpus : int
        number of gpus        
    """

    def __init__(self, theta, nz, n, pnz, center, ngpus):
        """Init the base class with allocating gpu memory in c++ context."""

        # create class for the tomo transform associated
        super().__init__(len(theta), pnz, n, center, cp.array(
            theta.astype('float32')).data.ptr, ngpus)

        self.nz = nz  # total number of slices
        self.filter = \
            {
                'none': 0,
                'ramp': 1,
                'shepp': 2,
                'hann': 3,
                'parzen': 4
            }

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def fwd_tomo_gpu(self, u, lock, ids):
        """Forward Radon transform of one z-slcie partition"""

        # wait for a nonbusy gpu and lock it for computations
        global BUSYGPUS
        lock.acquire()  # will block if lock is already held
        for k in range(self.ngpus):
            if BUSYGPUS[k] == 0:
                BUSYGPUS[k] = 1
                gpu = k
                break
        lock.release()
        cp.cuda.Device(gpu).use()

        # copy input function to gpu
        u_gpu = cp.array(u[ids])

        # process 2 float32 slices as 1 complex64
        d_gpu = cp.zeros([self.pnz, self.ntheta, self.n, 2], dtype='float32')
        u_gpuc = u_gpu[::2]+1j*u_gpu[1::2]
        # C++ wrapper, send pointers to GPU arrays
        self.fwd(d_gpu.data.ptr, u_gpuc.data.ptr, gpu)
        # reorder to float32
        d_gpu = np.moveaxis(d_gpu, 3, 1).reshape(
            2*self.pnz, self.ntheta, self.n)
        # copy output function to cpu
        d = d_gpu.get()

        # unlock gpu
        BUSYGPUS[gpu] = 0

        return d

    def forward(self, u):
        """Forward Radon transform by z-slice partitions"""

        # form a list of slice partitions
        ids_list = [None]*int(np.ceil(self.nz/float(2*self.pnz)))
        for k in range(0, len(ids_list)):
            ids_list[k] = range(k*2*self.pnz, min(self.nz, (k+1)*2*self.pnz))

        # init a global array of busy gpus, slice partitons are given to gpu whenever
        # it is not locked with computing other partition
        lock = threading.Lock()
        global BUSYGPUS
        BUSYGPUS = np.zeros(self.ngpus)

        d = np.zeros([self.nz, self.ntheta, self.n], dtype='float32')
        # parallel computing of data partitions by several gpus
        with cf.ThreadPoolExecutor(self.ngpus) as e:
            shift = 0
            for di in e.map(partial(self.fwd_tomo_gpu, u, lock), ids_list):
                d[np.arange(0, di.shape[0])+shift] = di
                shift += di.shape[0]
        # reorder to projection order
        d = d.swapaxes(0, 1)

        return d

    def adj_tomo_gpu(self, d, filter_id, lock, ids):
        """Inverse Radon transform of one z-slice partition"""

        # wait for a nonbusy gpu and lock it for computations
        global BUSYGPUS
        lock.acquire()  # will block if lock is already held
        for k in range(self.ngpus):
            if BUSYGPUS[k] == 0:
                BUSYGPUS[k] = 1
                gpu = k
                break
        lock.release()
        cp.cuda.Device(gpu).use()

        # reorder to sinogram order
        d = d.swapaxes(0, 1)
        # copy input function to gpu
        d_gpu = cp.array(d[ids])

        # reconstruct two float32 slices as 1 complex64
        u_gpu = cp.zeros([self.pnz, self.n, self.n, 2], dtype='float32')
        d_gpuc = d_gpu[::2]+1j*d_gpu[1::2]
        # C++ wrapper, send pointers to GPU arrays
        self.adj(u_gpu.data.ptr, d_gpuc.data.ptr, filter_id, gpu)
        # reorder to float32 array
        u_gpu = np.moveaxis(u_gpu, 3, 1).reshape(2*self.pnz, self.n, self.n)

        # copy output function to cpu
        u = u_gpu.get()

        # unlock gpu
        BUSYGPUS[gpu] = 0

        return u

    def recon(self, d, filter_type='none'):
        """Inverse Radon transform by z-slice partitions"""

        # form a list of slice partitions
        ids_list = [None]*int(np.ceil(self.nz/float(2*self.pnz)))
        for k in range(0, len(ids_list)):
            ids_list[k] = range(k*2*self.pnz, min(self.nz, (k+1)*2*self.pnz))

        # init a global array of busy gpus, slice partitons are given to gpu whenever
        # it is not locked with computing other partition
        lock = threading.Lock()
        global BUSYGPUS
        BUSYGPUS = np.zeros(self.ngpus)

        u = np.zeros([self.nz, self.n, self.n], dtype='float32')
        # parallel computing of data partitions by several gpus
        with cf.ThreadPoolExecutor(self.ngpus) as e:
            shift = 0
            for ui in e.map(partial(self.adj_tomo_gpu, d, self.filter[filter_type], lock), ids_list):
                u[np.arange(0, ui.shape[0])+shift] = ui
                shift += ui.shape[0]

        return u
