#include "radonusfft.cuh"
#include "kernels.cuh"

radonusfft::radonusfft(size_t ntheta, size_t pnz, size_t n, float center,
  size_t theta_, size_t ngpus): ntheta(ntheta), pnz(pnz), n(n), center(center), ngpus(ngpus) {

  float eps = 1e-2; // accuracy for usfft, gives interpolation radius in the frequency domain (m)
  mu = -log(eps) / (2 * n * n);
  m = ceil(2 * n * 1 / PI * sqrt(-mu * log(eps) + (mu * n) * (mu * n) / 4));

  // pointers to arrays on different gpus
  f = new float2 * [ngpus];
  g = new float2 * [ngpus];
  fdee = new float2 * [ngpus];
  x = new float * [ngpus];
  y = new float * [ngpus];
  shiftfwd = new float2 * [ngpus];
  shiftadj = new float2 * [ngpus];
  theta = new float * [ngpus];
  plan1d = new cufftHandle[ngpus];
  plan2d = new cufftHandle[ngpus];

  // memory allocation on each gpu
  for (int igpu = 0; igpu < ngpus; igpu++) {
    cudaSetDevice(igpu);
    cudaMalloc((void ** ) & f[igpu], n * n * pnz * sizeof(float2));
    cudaMalloc((void ** ) & g[igpu], n * ntheta * pnz * sizeof(float2));
    cudaMalloc((void ** ) & fdee[igpu],
      (2 * n + 2 * m) * (2 * n + 2 * m) * pnz * sizeof(float2));

    cudaMalloc((void ** ) & x[igpu], n * ntheta * sizeof(float));
    cudaMalloc((void ** ) & y[igpu], n * ntheta * sizeof(float));
    cudaMalloc((void ** ) & theta[igpu], ntheta * sizeof(float));
    cudaMemcpy(theta[igpu], (float * ) theta_, ntheta * sizeof(float), cudaMemcpyDefault);

    int ffts[2];
    int idist;
    int inembed[2];
    // fft 2d
    ffts[0] = 2 * n;
    ffts[1] = 2 * n;
    idist = (2 * n + 2 * m) * (2 * n + 2 * m);
    inembed[0] = 2 * n + 2 * m;
    inembed[1] = 2 * n + 2 * m;
    cufftPlanMany( & plan2d[igpu], 2, ffts, inembed, 1, idist, inembed, 1, idist,
      CUFFT_C2C, pnz);

    // fft 1d
    ffts[0] = n;
    idist = n;
    inembed[0] = n;
    cufftPlanMany( & plan1d[igpu], 1, ffts, inembed, 1, idist, inembed, 1, idist,
      CUFFT_C2C, ntheta * pnz);
    cudaMalloc((void ** ) & shiftfwd[igpu], n * sizeof(float2));
    cudaMalloc((void ** ) & shiftadj[igpu], n * sizeof(float2));

    // compute shifts with respect to the rotation center
    takeshift << < ceil(n / 1024.0), 1024 >>> (shiftfwd[igpu], -(center - n / 2.0), n);
    takeshift << < ceil(n / 1024.0), 1024 >>> (shiftadj[igpu], (center - n / 2.0), n);
  }

  //back current gpu to 0
  cudaSetDevice(0);

  // gpu grid and block sizes used in cuda kernels
  BS2d = dim3(32, 32);
  BS3d = dim3(32, 32, 1);

  GS2d0 = dim3(ceil(n / (float) BS2d.x), ceil(ntheta / (float) BS2d.y));
  GS3d0 = dim3(ceil(n / (float) BS3d.x), ceil(n / (float) BS3d.y),
    ceil(pnz / (float) BS3d.z));
  GS3d1 = dim3(ceil(2 * n / (float) BS3d.x), ceil(2 * n / (float) BS3d.y),
    ceil(pnz / (float) BS3d.z));
  GS3d2 = dim3(ceil((2 * n + 2 * m) / (float) BS3d.x),
    ceil((2 * n + 2 * m) / (float) BS3d.y), ceil(pnz / (float) BS3d.z));
  GS3d3 = dim3(ceil(n / (float) BS3d.x), ceil(ntheta / (float) BS3d.y),
    ceil(pnz / (float) BS3d.z));

  is_free = false;
}

// destructor, memory deallocation
radonusfft::~radonusfft() {
  free();
}

void radonusfft::free() {
  if (!is_free) {
    for (int igpu = 0; igpu < ngpus; igpu++) {
      cudaSetDevice(igpu);
      cudaFree(f[igpu]);
      cudaFree(g[igpu]);
      cudaFree(fdee[igpu]);
      cudaFree(x[igpu]);
      cudaFree(y[igpu]);
      cudaFree(shiftfwd[igpu]);
      cudaFree(shiftadj[igpu]);
      cufftDestroy(plan2d[igpu]);
      cufftDestroy(plan1d[igpu]);
    }
    cudaFree(f);
    cudaFree(g);
    cudaFree(fdee);
    cudaFree(x);
    cudaFree(y);
    cudaFree(shiftfwd);
    cudaFree(shiftadj);
    is_free = true;
  }
}

void radonusfft::fwd(size_t g_, size_t f_, size_t igpu) { // Forward radon transform on a gpu

  cudaSetDevice(igpu);

  // copy object to preallocated array on gpu
  cudaMemcpy(f[igpu], (float2 * ) f_, n * n * pnz * sizeof(float2), cudaMemcpyDefault);
  cudaMemset(fdee[igpu], 0, (2 * n + 2 * m) * (2 * n + 2 * m) * pnz * sizeof(float2));

  // unequally spaced coordinates in frequencies
  takexy << < GS2d0, BS2d >>> (x[igpu], y[igpu], theta[igpu], n, ntheta);

  // division by kernel function
  divphi << < GS3d2, BS3d >>> (fdee[igpu], f[igpu], mu, n, pnz, m, TOMO_FWD);

  // 2D fft of the object
  fftshiftc2d << < GS3d2, BS3d >>> (fdee[igpu], 2 * n + 2 * m, pnz);
  cufftExecC2C(plan2d[igpu], (cufftComplex * ) & fdee[igpu][m + m * (2 * n + 2 * m)],
    (cufftComplex * ) & fdee[igpu][m + m * (2 * n + 2 * m)], CUFFT_FORWARD);
  fftshiftc2d << < GS3d2, BS3d >>> (fdee[igpu], 2 * n + 2 * m, pnz);
  // wrap boundaries
  wrap << < GS3d2, BS3d >>> (fdee[igpu], n, pnz, m, TOMO_FWD);

  // gathering (interpolation) to unequally spaced grid in frequencies
  gather << < GS3d3, BS3d >>> (g[igpu], fdee[igpu], x[igpu], y[igpu], m, mu, n, ntheta, pnz, TOMO_FWD);

  // shift with respect to given center
  shift << < GS3d3, BS3d >>> (g[igpu], shiftfwd[igpu], n, ntheta, pnz);
  // 1D ifft on the projections
  fftshiftc1d << < GS3d3, BS3d >>> (g[igpu], n, ntheta, pnz);
  cufftExecC2C(plan1d[igpu], (cufftComplex * ) g[igpu], (cufftComplex * ) g[igpu], CUFFT_INVERSE);
  fftshiftc1d << < GS3d3, BS3d >>> (g[igpu], n, ntheta, pnz);

  // copy result to output array
  cudaMemcpy((float2 * ) g_, g[igpu], n * ntheta * pnz * sizeof(float2), cudaMemcpyDefault);
}

void radonusfft::adj(size_t f_, size_t g_, size_t filterid, size_t igpu) { //Adjoint Radon transform on a gpu
  cudaSetDevice(igpu);

  // copy projections to preallocated array on gpu  
  cudaMemcpy(g[igpu], (float2 * ) g_, n * ntheta * pnz * sizeof(float2), cudaMemcpyDefault);
  cudaMemset(fdee[igpu], 0, (2 * n + 2 * m) * (2 * n + 2 * m) * pnz * sizeof(float2));

  // unequally spaced coordinates in frequencies
  takexy << < GS2d0, BS2d >>> (x[igpu], y[igpu], theta[igpu], n, ntheta);

  // 1D fft of the projections    
  fftshiftc1d << < GS3d3, BS3d >>> (g[igpu], n, ntheta, pnz);
  cufftExecC2C(plan1d[igpu], (cufftComplex * ) g[igpu], (cufftComplex * ) g[igpu], CUFFT_FORWARD);
  fftshiftc1d << < GS3d3, BS3d >>> (g[igpu], n, ntheta, pnz);

  // filter projections
  if (filterid != 0)
    applyfilter << < GS3d3, BS3d >>> (g[igpu], filterid, n, ntheta, pnz);

  // shift with respect to given center
  shift << < GS3d3, BS3d >>> (g[igpu], shiftadj[igpu], n, ntheta, pnz);

  // gathering(intepolation) from unequally spaced grid in frequencies
  gather << < GS3d3, BS3d >>> (g[igpu], fdee[igpu], x[igpu], y[igpu], m, mu, n, ntheta, pnz, TOMO_ADJ);
  wrap << < GS3d2, BS3d >>> (fdee[igpu], n, pnz, m, TOMO_ADJ);

  // 2D ifft of the object
  fftshiftc2d << < GS3d2, BS3d >>> (fdee[igpu], 2 * n + 2 * m, pnz);
  cufftExecC2C(plan2d[igpu], (cufftComplex * ) & fdee[igpu][m + m * (2 * n + 2 * m)],
    (cufftComplex * ) & fdee[igpu][m + m * (2 * n + 2 * m)], CUFFT_INVERSE);
  fftshiftc2d << < GS3d2, BS3d >>> (fdee[igpu], 2 * n + 2 * m, pnz);

  // division by the kernel function
  divphi << < GS3d0, BS3d >>> (fdee[igpu], f[igpu], mu, n, pnz, m, TOMO_ADJ);

  // copy result to output array
  cudaMemcpy((float2 * )f_, f[igpu], n * n * pnz * sizeof(float2),
    cudaMemcpyDefault);
}