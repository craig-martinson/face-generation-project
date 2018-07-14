# Generative Adversarial Networks (GANs)

Generative Adversarial Networks (GANs) project developed for Udacity's Deep Learning Nanodegree. The goal of this project is to use generative adversarial networks to generate new images of faces.

## Getting Started

### Setup Environment

#### Clone the Repository

``` batch
git clone https://github.com/craig-martinson/face-generation-project.git
cd face-generation-project
```

#### Setup Linux

Tested on the following environment:

- Ubuntu 16.04.4 LTS
- NVIDIA GTX1080 (driver version 384.130)
- CUDA® Toolkit 9.0
- cuDNN v7.0

Create a Linux Conda environment with **CPU** backend and upgrade tensorflow:

``` batch
conda create --name face-generation-project pip python=3.6 numpy jupyter tqdm
conda activate face-generation-project
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl
python -m ipykernel install --user --name face-generation-project --display-name "face-generation-project"
 ```

Create a Linux Conda environment with **GPU** backend and upgrade tensorflow:

``` batch
conda create --name face-generation-project pip python=3.6 numpy jupyter tqdm
conda activate face-generation-project
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.8.0-cp36-cp36m-linux_x86_64.whl
python -m ipykernel install --user --name face-generation-project --display-name "face-generation-project"
```

#### Setup Windows

Tested on the following environment:

- Windows 10 Pro, 64-bit
- NVIDIA GTX1080 (driver version 385.54)
- CUDA® Toolkit 9.0
- cuDNN v7.0

Create a Windows Conda environment with **CPU** backend and upgrade tensorflow:

``` batch
conda create --name face-generation-project pip python=3.6 numpy jupyter tqdm tensorflow
conda activate face-generation-project
python -m ipykernel install --user --name face-generation-project --display-name "face-generation-project"
 ```

Create a Windows Conda environment with **GPU** backend and upgrade tensorflow:

``` batch
conda create --name face-generation-project pip python=3.6 numpy jupyter tqdm tensorflow-gpu
conda activate face-generation-project
python -m ipykernel install --user --name face-generation-project --display-name "face-generation-project"
```

#### Setup macOS

Tested on the following environment:

- macOS High Sierra

Create a macOS Conda environment with **CPU** backend and upgrade tensorflow:

``` batch
conda create --name face-generation-project pip python=3.6 numpy jupyter tqdm
conda activate face-generation-project
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py3-none-any.whl
python -m ipykernel install --user --name face-generation-project --display-name "face-generation-project"
 ```

## Jupyter Notebooks

The following jupyter notebooks were developed to support this project:

Description | Link
--- | ---
Project notebook provided by Udacity, demonstrates GANs with TensorFlow | [Face Generation Notebook](./dlnd_face_generation.ipynb)

## References

The following resources were used in developing this project:

- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)
