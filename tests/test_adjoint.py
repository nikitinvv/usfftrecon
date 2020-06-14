import numpy as np
import usfftrecon as ur

if __name__ == "__main__":
# Check adjoint test <R*Rf,g>=<Rf,Rf>

    # Model parameters
    n = 256  # object size in x,y
    nz = 256  # object size in z
    ntheta = 500  # number of angles (rotations)
    center = n/2  # rotation center
    theta = np.linspace(0, np.pi, ntheta).astype('float32')  # angles
    pnz = 32  # number of complex64 slices in each partition for simultaneous processing in tomography, 
              # should be less or equal than (nz/ngpus/2) because each 2 float32 slices are coupled into 1 complex64

    # init object
    u0 = np.random.random([nz, n, n]).astype('float32')/n/ntheta
    ngpus = 1
    # Class gpu solver
    with ur.SolverTomo(theta, nz, n, pnz, center, ngpus) as slv:
        # forward transform
        data = slv.forward(u0)
        # adjoint transform
        u1 = slv.recon(data)

        # adjoint test
        t1 = np.sum(data*np.conj(data))
        t2 = np.sum(u0*np.conj(u1))
        print(f"Adjoint test: {t1.real:06f}{t1.imag:+06f}j "
              f"=? {t2.real:06f}{t2.imag:+06f}j")
        np.testing.assert_allclose(t1, t2, rtol=1e-4)
