## Simple C++ Torch tutorial with Go binding

### Pre-requisites

This framework was tested on two different architectures, each of which requires its own specific development tool suite.

#### MacOS arm64 MPS

The MacOS code has been tested on MacOS arm64 (MPS) Sequoia, with command line xcode installed including _make_, _clang v16.0.0_.

* Install xcode CLI development tools from apple:

```
xcode-select --install
```

* Install [brew](https://brew.sh/), python3.12 or higher, and cmake

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python3.12
brew install cmake
```

#### Linux amd64 CUDA

This has been tested on Linux Ubuntu 24.04 with CUDA 12.2 and Nvidia A100.
The provided script [linux-setup.sh](./linux-setup.sh) automates the steps outlined below for this specific configuration and can serve as a starting point for other configurations.

You need _g++-12, build-essential,_ and _libxml2_. Note that CUDA Toolkit 12.2 will not work with a higher verison of g++ if using _cmake_.

* Install python and development tools

```
sudo apt update
sudo apt install -y  wget g++-12 build-essential git libxml2 unzip cmake python3.12 python3.12-venv
sudo apt autoremove -y
sudo apt -y clean
```

* Verify your CUDA driver and CUDA Toolkit version (_sudo apt install pciutils_ if you don't have _lspci_):

```
lspci | grep -i nvidia
nvidia-smi
nvcc --version
```

If _nvcc_ is not found, you must install a version of CUDA tooklkit from [nvidia developer](https://developer.nvidia.com/cuda-downloads/)
that is compatible with your CUDA driver.

### Install LibTorch++

LibTorch is required in order to compile a C++ application.

Two options are possible:

#### Libtorch++

Extract the appropriate **LibTorch** C++ zip from https://pytorch.org/ to a folder, e.g. _/usr/local/libtorch_.

If you intend to use _cmake_, verify that _CMAKE_PREFIX_PATH_ in all _*/CMakeLists.txt_ is properly set to include the home folder of the libtorch library.


#### PyTorch's Libtorch++

Create the Python virtual environment and install PyTorch inside the virtual environment using pip:

```
cd python
python -m venv venv
. venv/bin/activate
pip install torch
```

If you intend to use _cmake_, verify that _CMAKE_PREFIX_PATH_ in all _*/CMakeLists.txt_ is properly set to include the home folder of the libtorch library, typically under _lib/python3.*/site-packages/torch_ in the venv folder.

Note on C++ ABI Compatibility:
The version of PyTorch may have been compiled with the older C++ ABI, in which case you will get undefined symbols with
_cxx11_ during the compilation. If that happens, you must explicitly add the _-D_GLIBCXX_USE_CXX11_ABI=0_ flag to _CCFLAGS_ when building with make.
This is to ensures that the code is compiled using the same ABI as the PyTorch libraries.

