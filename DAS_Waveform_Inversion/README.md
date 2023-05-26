# TorchFWI-DAS
**Note**: This version code is modified by Haipeng Li, and the original code is developed by Dongzhuo Li at Stanford. The original code is available at https://github.com/lidongzh/TorchFWI/tree/master. If the user encounter any issues or have further questions, feel free to reach out to me at haipeng@stanford.edu for the modification. The current version is modifed to enable the inversion of the DAS data (only support the either horizontal or vertical straight fiber for now).

TorchFWI is an elastic full-waveform inversion (FWI) package integrated with the deep-learning framework PyTorch. On the one hand, it enables the integration of FWI with neural networks and makes it easy to create complex inversion workflows. On the other hand, the multi-GPU-accelerated FWI component with a boundary-saving method offers high computational efficiency. One can use the suite of built-in optimizers in PyTorch or Scipy (e.g., L-BFGS-B) for inversion.


## Installation

To run the code, you'll need to set up a Conda environment named **torchfwi** and install the required packages. Follow the steps below:

1. Install Conda: If you don't have Conda installed, you can download and install Miniconda or Anaconda from the official website ([Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) | [Anaconda](https://www.anaconda.com/products/individual)).

2. Open a terminal or command prompt.

3. Create the Conda environment:
```bash
conda create -n torchfwi python=3.9
```

4. Activate the Conda environment:
```bash
conda activate torchfwi
```
5. Set up the environment:
```bash
# In some clusters, there are installed modules, one can load them by
module load cuda/11.3.1    
module load gcc/9.1.0      # the gcc version should be 5.0.0 <= V <= 10.0.0
module load ninja/1.9.0
```

If these modules are not installed, one can install them one by one:

**Install gcc**: further refer to https://gcc.gnu.org/install/binaries.html

**Install cuda**: further refer to https://developer.nvidia.com/cuda-toolkit-archive

**Install ninja**: further refer to: https://github.com/ninja-build/ninja/releases


6. Install the required packages:
```bash
# Install PyTorch with GPU support
pip install torch

# Install the remaining required packages
pip install numpy scipy matplotlib ninja
```

The above steps have been successfully tested on CentOS-7 from scratch. 

## Usage
To run the code, open the following jupyter notebook under the folder "notebooks" and run all the cells:
```bash
./notebooks/001-FWI-Anomaly-Vp-Vs-Den.ipynb
./notebooks/002-FWI-Anomaly-Lame-Den.ipynb
./notebooks/003-FWI-Anomaly-IP-IS-Den.ipynb
./notebooks/004-FWI-Rock-Physics.ipynb
```
These notebooks are self-contained and contains the code to generate the synthetic data, and run the inversion, as well as the code to plot the results. Please be aware that the path in the above notebooks are set to the path in my computer, one need to change the path to one's own path.


This package uses just-in-time (JIT) compilation for the FWI code. It only compiles the first time you run the code, and no explicit ''make'' is required. An NVIDIA cuda compiler is needed.

If you find this package helpful in your research, please kindly cite
1. **Dongzhuo Li**, **Kailai Xu**, Jerry M. Harris, and Eric Darve. [*Time-lapse Full-waveform Inversion for Subsurface Flow Problems with Intelligent Automatic Diﬀerentiation*](https://arxiv.org/abs/1912.07552).
2. **Dongzhuo Li**, **Jerry M. Harris**, **Biondo Biondi**, and **Tapan Mukerji**. 2019. [*Seismic full waveform inversion: nonlocal similarity, sparse dictionary learning, and time-lapse inversion for subsurface flow.*](http://purl.stanford.edu/ds556fq6692).
3. **Kailai Xu**, **Dongzhuo Li**, Eric Darve, and Jerry M. Harris. [*Learning Hidden Dynamics using Intelligent Automatic Differentiation*](http://arxiv.org/abs/1912.07547).
