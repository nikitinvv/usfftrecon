import numpy as np
import usfftrecon as pt
from timing import *
if __name__ == "__main__":
# Performance tests on several gpus    

    # Model parameters
    n = 2448  # object size in x,y
    nz = 2048  # object size in z
    ntheta = 1500  # number of angles (rotations)
    center = n/2  # rotation center
    theta = np.linspace(0, np.pi, ntheta).astype('float32')  # angles
    pnz = 16  # number of complex64 slices in each partition for simultaneous processing on each gpu, 
              # should be less or equal than (nz/ngpus/2) because each 2 float32 slices are coupled into 1 complex64
    # init some data
    data = np.zeros([ntheta,nz,n],dtype='float32')
    data[:,:,n//4:3*n//4]=1
    
    # check times for different number of gpus
    for ngpus in range(1,5):
        # Class gpu solver
        with pt.SolverTomo(theta, nz, n, pnz, center, ngpus) as slv:
            t = 0
            for k in range(3):
                tic()        
                u = slv.recon(data,'hann')
                t += toc()
            print('data shape:',data.shape,'number of gpus:',ngpus,'rec norm:',np.linalg.norm(u),'rec time:', t/3)       
            
