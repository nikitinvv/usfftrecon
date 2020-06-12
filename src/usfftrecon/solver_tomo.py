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
    Attribtues
    ----------
    ntheta : int
        The number of projections.    
    n, nz : int
        The pixel width and height of the projection.
    pnz : int
        The number of slice partitions to process together
        simultaneously.
    ngpus : int
        Number of gpus        
    """

    def __init__(self, theta, ntheta, nz, n, pnz, center, ngpus):
        """Please see help(SolverTomo) for more info."""
        # create class for the tomo transform associated with first gpu
        super().__init__(ntheta, pnz, n, center, theta.ctypes.data, ngpus)
        self.nz = nz        
        
    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def fwd_tomo(self, u, gpu):
        """Radon transform (R)"""
        res = cp.zeros([self.ntheta, self.pnz, self.n], dtype='complex64')
        u0 = u.astype('complex64')
        # C++ wrapper, send pointers to GPU arrays
        self.fwd(res.data.ptr, u0.data.ptr, gpu)        
        return res.real

    def adj_tomo(self, data, gpu):
        """Adjoint Radon transform (R^*)"""
        res = cp.zeros([self.pnz, self.n, self.n], dtype='complex64')
        data0 = data.astype('complex64')
        # C++ wrapper, send pointers to GPU arrays        
        self.adj(res.data.ptr, data0.data.ptr, gpu)
        return res.real    

    def fwd_tomo_part(self,u,lock,ids):
        """Forward Radon transform of a slice partition"""

        global BUSYGPUS
        lock.acquire()  # will block if lock is already held
        for k in range(self.ngpus):
            if BUSYGPUS[k] == 0:
                BUSYGPUS[k] = 1
                gpu = k
                break
        lock.release()

        cp.cuda.Device(gpu).use()
        u_gpu = cp.array(u[ids])
        # reconstruct
        d_gpu = self.fwd_tomo(u_gpu, gpu)
        d = d_gpu.get()

        BUSYGPUS[gpu] = 0

        return d
  
    
    def fwd_tomo_batch(self, u):
        """Forward Radon transform by z-slice partitions"""
        d = np.zeros([self.ntheta,self.nz,self.n],dtype='float32')
        
        ids_list = [None]*int(np.ceil(self.nz/float(self.pnz)))
        for k in range(0, len(ids_list)):
            ids_list[k] = range(k*self.pnz, min(self.nz, (k+1)*self.pnz))
        
        lock = threading.Lock()
        global BUSYGPUS
        BUSYGPUS = np.zeros(self.ngpus)
        with cf.ThreadPoolExecutor(self.ngpus) as e:
            shift = 0
            for di in e.map(partial(self.fwd_tomo_part, u, lock), ids_list):
                d[:,np.arange(0, di.shape[1])+shift] = di
                shift += di.shape[1]
        cp.cuda.Device(0).use()      

        return d

    def adj_tomo_part(self,d,lock,ids):
        """Adjoint Radon transform (with possible filter) of a slice partition"""

        global BUSYGPUS
        lock.acquire()  # will block if lock is already held
        for k in range(self.ngpus):
            if BUSYGPUS[k] == 0:
                BUSYGPUS[k] = 1
                gpu = k
                break
        lock.release()

        cp.cuda.Device(gpu).use()
        d_gpu = cp.array(d[:, ids])
        # reconstruct
        u_gpu = self.adj_tomo(d_gpu, gpu)
        u = u_gpu.get()

        BUSYGPUS[gpu] = 0

        return u
  
    
    def adj_tomo_batch(self, d):
        """Adjoint Radon transform (with possible filter)  by z-slice partitions"""
        u = np.zeros([self.nz,self.n,self.n],dtype='float32')
        ids_list = [None]*int(np.ceil(self.nz/float(self.pnz)))
        for k in range(0, len(ids_list)):
            ids_list[k] = range(k*self.pnz, min(self.nz, (k+1)*self.pnz))
        
        lock = threading.Lock()
        global BUSYGPUS
        BUSYGPUS = np.zeros(self.ngpus)
        with cf.ThreadPoolExecutor(self.ngpus) as e:
            shift = 0
            for ui in e.map(partial(self.adj_tomo_part, d, lock), ids_list):
                u[np.arange(0, ui.shape[0])+shift] = ui
                shift += ui.shape[0]
        cp.cuda.Device(0).use()      

        return u


 