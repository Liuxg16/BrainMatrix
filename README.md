We improve the notable project MXnet and name the new version as BrainMatrix.
after gitting clone this project,just follow the step to compile:
1.vim the comfig.mk
    if your machine are equiped with gpu, no edit
    if your machine are not equiped with gpu, change: 
            USE_CUDA = 0 
            delete:USE_CUDA_PATH = /usr/local/cuda
1. make j4 #4 is the number of kernels in cpu
