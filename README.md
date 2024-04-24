# cluster-turbulence

# Using GPUs on Newton

Steps to get paicos working on Newton, including GPU-code.

## Option 1 (running from terminal)

### Install Miniforge3
```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Full installation instructions here: https://github.com/conda-forge/miniforge/?tab=readme-ov-file#download

### Make sure that your conda install is working before proceeding, then install paicos

```
conda create -q -n paicos-conda python=3.11 --yes
conda activate paicos-conda
conda install paicos --yes
conda install pytest pytest-order cython ipython --yes
conda install -c numba cupy cudatoolkit=11.7
```

### Start an interactive gpu-session
```
srun -p a100 --gres=gpu:1 -n 1 -c 1 --time=1:00:00 --mem=80gb --pty /bin/bash
```

### Activate the conda environment
```
conda activate paicos-conda
```

### Check that it is working
```
python -c "import paicos; paicos.gpu_init()"
```

## Option 2 (run a notebook)

First follow the installation instructions above, then
```
conda activate paicos-conda
conda install jupyter
```

Now get a GPU
```
srun -p a100 --gres=gpu:1 -n 1 -c 1 --time=1:00:00 --mem=80gb --pty /bin/bash 
```
Start the jupyter server on the GPU:
```
conda activate paicos-conda
jupyter notebook --no-browser 
```

Check which GPU you are on (in this example it is ngpu050), then SSH onto it (port numbers will likely differ in your case)
```
ssh  -L 2012:localhost:2012 -J  berlok@login.aip.de,berlok@obelisk,berlok@nnewl3.cls ngpu050
```

Now open the jupyter server in a browser, e.g., go to http://127.0.0.1:2012/tree
Then you can hopefully make a notebook that looks like this:



