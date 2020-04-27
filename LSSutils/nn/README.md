# Neural Network Modeling of Imaging Systematics

Required packages:
* python > 3.7
* pytorch

## Installation
1. Install Conda by following the instructions at [docs.conda.io](https://docs.conda.io/projects/conda/en/latest/index.html). E.g.,
```
$> wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$> sha256sum Miniconda3-latest-Linux-x86_64.sh
```
2. Update conda with `$> conda update conda`
3. Create a fresh environment called *sysnet* (it’s up to you!)
```
$> conda create -n sysnet
```
4. Activate the environment, and install pytorch
```
$> conda activate sysnet
$> conda install pytorch torchvision -c pytorch
```
5. Install the rest of the packages
```
$> conda install ipykernel jupyter matplotlib pandas
```
6. Add the kernel to Jupyter
```
$> python -m ipykernel install --user --name=sysnet --display-name "python (sysnet)"
```
