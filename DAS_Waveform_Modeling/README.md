## DAS Waveform Modeling Toolbox
This repository contains the code for modeling the elastic wavefield propagation in isotropic media using the first-order velocity-stress formulation. The staggered-grid finite-difference method is used to discretize the governing equations. The code is written in Python, and the jit compiler from the Numba package is utilized to accelerate the computation.

Also this repository contain the analytical displacement solution for the 3D/2D wave equation due to a moment tensor source in a homogeneous, isotropic elastic medium based on Aki & Richards (2002). The analytical solutions are used to bench mark the numeircal results. 


## Installation Tutorial

To run the code, you'll need to set up a Conda environment named **torchfwi** and install the required packages. Follow the steps below:

1. Install Conda: If you don't have Conda installed, you can download and install Miniconda or Anaconda from the official website ([Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) | [Anaconda](https://www.anaconda.com/products/individual)).

2. Open a terminal or command prompt.

3. Create the Conda environment:
```bash
conda create -n das python=3.9
```

4. Activate the Conda environment:
```bash
conda activate das
```

5. Install the required packages:
```bash
# use pip
pip install multiprocessing numba numpy matplotlib tqdm joblib

# or use conda
conda install -c conda-forge multiprocessing numba numpy matplotlib tqdm joblib
```
Make sure to create and activate a Python environment before installing the packages to keep your dependencies isolated and organized.


## Usage
To use this code, follow these steps:

Run **./matlab/DAS_Geometry_Homogeneous.m** with MATLAB to generate the DAS system's geometry, including the sensor locations based on the gauge length, and its sensitivity matrix.

Run **./matlab/DAS_Geometry_Overthrust.m** with MATLAB to generate the DAS system's geometry, including the sensor locations based on the gauge length, and its sensitivity matrix.

Run **./notebooks/000-Solver-Benchmark.ipynb** to benchmark the forward code for modeling geophone data and strain data against the analytical solution. It provides accurate results.

Run **./notebooks/Fig-2-3-Analytical-DAS-Waveform.ipynb** to explore the forward code (analytical solution) for modeling DAS data in 3D homogeneous model.

Run **./notebooks/Fig-4-Numerical-DAS-Waveform.ipynb** to explore the forward code (finite-difference solution) for modeling DAS data in a part of the Overthrust model.

If you encounter any issues or have further questions, feel free to reach out to me at haipeng@stanford.edu

