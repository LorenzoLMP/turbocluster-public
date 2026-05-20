# How to install

## Prerequisites

Turbocluster is a library that uses CUDA kernels written in Python with Cupy and Numba to accelerate filtering of physical fields. Therefore it is **required** that your environment has a recent Nvidia GPU with a CUDA toolkit. For futher information, refer to the Cupy installation [here](https://docs.cupy.dev/en/stable/install.html).

Note: for Cupy and Numba to correctly locate your CUDA installation you may have to set the environment variables CUDA_HOME and CUDA_PATH in your .bashrc or .bash_profile as follows
(substitute with the path to the CUDA installation on your system, and replace version number as necessary):

```
export CUDA_HOME=/path/to/cuda/12.0 # numba
export CUDA_PATH=/path/to/cuda/12.0 # cupy
```

## Installation steps

1. Create a conda environment:

    - Download miniforge and follow instructions at <https://github.com/conda-forge/miniforge/#download>
    - Create an environment called e.g. `turbocluster': ```conda create --name turbocluster python=3.14```
    - Install pip ```conda install pip```

2. Clone the directory from GitHub and install the required packages:

    ```
    git clone git@github.com:LorenzoLMP/turbocluster-public.git
    cd turbocluster-public
    pip install -r requirements.txt
    ```

3. Add the directory of the turbocluster repository to your PYTHONPATH, e.g., in the `.bash_profile`:
    ```
    export PYTHONPATH=$PYTHONPATH:/path/to/turbocluster-public
    ```

4. Open a new shell. Check that the installation worked and that you can import turbocluster: 
    ```
    python -c "import turbocluster"
    python -c "import cupy"
    python -c "from numba import cuda"
    python -c "import paicos; paicos.gpu_init()"
    ```
    If no error message appeared, you are good to go!
