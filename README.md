# prefect-cellpose
A collection of prefect flows to run cellpose.

## Installation
Create a new conda environment and install the requirements.
```shell
conda env create -f environment.yaml
```

Or follow the official install instructions from [cellpose](https://github.com/MouseLand/cellpose#instructions).

__Note:__ Cellpose downloads and caches pre-trained models in your home-directory. [Check this out](https://cellpose.readthedocs.io/en/latest/installation.html#built-in-model-directory) if you want to change this behaviour.
```shell
export CELLPOSE_LOCAL_MODELS_PATH=/home/tibuch/Gitrepos/prefect-cellpose/models
```
