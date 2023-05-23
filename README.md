## SEP-2023: Elastic Full-Waveform Inversion of Distributed Acoustic Sensing Data: Modeling, Inversion, and Parameterization by Haipeng Li \& Biondo L. Biondi

### DAS Waveform Modeling
=====================
This repository contains the code for modeling the elastic wavefield propagation in isotropic media using the first-order velocity-stress formulation. The staggered-grid finite-difference method is used to discretize the governing equations. The code is written in Python, and the jit compiler from the Numba package is utilized to accelerate the computation.

Also this repository contain the analytical displacement solution for the 3D/2D wave equation due to a moment tensor source in a homogeneous, isotropic elastic medium based on Aki & Richards (2002). The analytical solutions are used to bench mark the numeircal results.

Please refer to the readme file in the subfolder for more details, including the installation tutorial and usage.


### DAS Waveform Inversion
======================
This repository contains the code for the DAS waveform inversion. The code is written in Python with GPU propagtors, and the package PyTorch is utilized to facilitate the computation. 

TorchFWI is an elastic full-waveform inversion (FWI) package integrated with the deep-learning framework PyTorch. On the one hand, it enables the integration of FWI with neural networks and makes it easy to create complex inversion workflows. On the other hand, the multi-GPU-accelerated FWI component with a boundary-saving method offers high computational efficiency. One can use the suite of built-in optimizers in PyTorch or Scipy (e.g., L-BFGS-B) for inversion.

**Note**: This version code is modified by Haipeng Li, and the original code is developed by Dongzhuo Li at Stanford. The original code is available at https://github.com/lidongzh/TorchFWI/tree/master. If the user encounter any issues or have further questions, feel free to reach out to me at haipeng@stanford.edu for the modification. The current version is modifed to enable the inversion of the DAS data (only support the either horizontal or vertical straight fiber for now).

Please refer to the readme file in the subfolder for more details, including the installation tutorial and usage.
