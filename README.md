# usfftrecon
multiGPU Tomography solver with the USFFT based method

## Installation from source

Solve dependencies in requirements.txt, then

python setup.py install

## 

## Tests
Check folder tests/:

1) test_adjoint.py - the adjoint test to check forward and inverse operators without filtering
2) test_recon.py - reconstruction of experimental data 
3) test_perf.py - check performance for different gpu numbers


## Performance listing
See folder perf/
