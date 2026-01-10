<h1 align="center">Sparfels: Fast Reconstruction from Sparse Unposed Imagery</h1>

<p align="center">
  <a href="https://scholar.google.com/citations?user=COw5y4EAAAAJ&hl=en"><strong>Shubhendu Jena</strong></a><sup>†</sup>
  ·
  <a href="https://ouasfi.github.io/"><strong>Amine Ouasfi</strong></a><sup>†</sup>
  ·
  <a href="https://orcid.org/0000-0002-4831-3343"><strong>Mae Younes</strong></a>
  ·
  <a href="https://boukhayma.github.io/"><strong>Adnane Boukhayma</strong></a>
</p>

<p align="center">
  <a href="https://www.inria.fr/en"> Inria </a>  
  <br>
  <strong>ICCV 2025</strong>
</p>

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue)](https://shubhendu-jena.github.io/Sparfels-web/)
&nbsp;
[![ArXiv Paper](https://img.shields.io/badge/ArXiv-Paper-red?style=flat&logo=arXiv&logoColor=red)](https://arxiv.org/abs/2505.02178)
&nbsp;
[![CVF Open Access](https://img.shields.io/badge/CVF-Open%20Access-blue?style=flat&logo=openaccess&logoColor=white)](https://openaccess.thecvf.com/content/ICCV2025/papers/Jena_Sparfels_Fast_Reconstruction_from_Sparse_Unposed_Imagery_ICCV_2025_paper.pdf)

</div>

<p align="center">
  <sup>†</sup> Equal contribution
</p>

<p align="center">
  <img src="assets/teaser.png" alt="Sparfels Teaser Figure" width="98%">
</p>

---

## Installation

> **Tested:** Ubuntu 22.04+, Conda, Python 3.9, PyTorch 2.5.1 (CUDA 12.1 wheels).

```bash
# Clone repo
git clone --recurse-submodules https://github.com/Shubhendu-Jena/Sparfels.git

# Create and activate environment
conda create -n sparfels python=3.9 -y
conda activate sparfels

# PyTorch (CUDA 12.1 wheels)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install kornia plyfile open3d scikit-image

# Local submodules
pip install submodules/simple-knn
pip install submodules/diff-surfel-rasterization

# Dust3R / Mast3R requirements
cd submodules/dust3r/ && pip install -r requirements.txt && pip install -r requirements_optional.txt || true
cd ../mast3r/ && pip install -r requirements.txt
cd ../..

# FAISS + Cython
conda install -y -c conda-forge "faiss-cpu=1.8.*"
pip install cython

# ASMK (build from source)
git clone https://github.com/jenicek/asmk
cd asmk/cython/ && cythonize *.pyx
cd ..
pip install .   # or: python setup.py build_ext --inplace
cd ..

# Build Dust3R curope extension
cd submodules/dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../../

# Extras + scientific stack pins
pip install mediapy embreex evo
pip install --upgrade "numpy>=2.0,<3" "scipy>=1.13,<2" "scikit-learn>=1.4,<2" "open3d>=0.18.0"

# PyTorch3D (for torch 2.5.1 + cu121)
conda install -y -c iopath iopath
conda install -y -c bottler nvidiacub
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+5043d15pt2.5.1cu121
