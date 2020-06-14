import dxchange
import numpy as np
import usfftrecon as pt
from timing import *

if __name__ == "__main__":
# Example of experimental data recontruction 
    # Number of gpus
    ngpus = 1        
    # Model parameters
    n = 2048  # object size in x,y
    nz = 8  # object size in z
    ntheta = 1501  # number of angles (rotations)
    center = 1024  # rotation center
    theta = np.linspace(0, np.pi, ntheta).astype('float32')  # angles
    pnz = 2  # number of complex64 slices in each partition for simultaneous processing in tomography, 
              # should be less or equal than (nz/ngpus/2) because each 2 float32 slices are coupled into 1 complex64
    # Load object
    data = dxchange.read_tiff('data/tomo_00001_proj.tiff').astype('float32')
    print(data.shape)

    # Class gpu solver
    with pt.SolverTomo(theta, nz, n, pnz, center, ngpus) as slv:
        tic()        
        u = slv.recon(data,'hann')
        print('time:', toc())       
        dxchange.write_tiff(u,'rec/rec_00001.tiff',overwrite=True)        
        print(np.linalg.norm(u))
