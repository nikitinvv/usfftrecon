import dxchange
import numpy as np
import usfftrecon as pt
from timing import *
import scipy.ndimage as ndimage
if __name__ == "__main__":
    
    # Model parameters
    n = 2448  # object size in x,y
    nz = n  # object size in z
    ntheta = 1500  # number of angles (rotations)
    center = n/2  # rotation center
    theta = np.linspace(0, np.pi, ntheta).astype('float32')  # angles
    pnz = 12  # number of complex64 slices in each partition for simultaneous processing on each gpu, 
              # should be less or equal than (nz/ngpus/2) because each 2 float32 slices are coupled into 1 complex64
    # Load object
    data = np.zeros([ntheta,nz,n],dtype='float32')
    data[:,:,n//4:3*n//4]=1
    for ngpus in range(1,9):
        # Class gpu solver
        with pt.SolverTomo(theta, nz, n, pnz, center, ngpus) as slv:
            tic()        
            u = slv.recon(data,'hann')
            print('data shape:',data.shape,'number of gpus:',ngpus,'rec norm:',np.linalg.norm(u),'rec time:', toc())       
            