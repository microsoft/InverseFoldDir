FROM mcr.microsoft.com/mirror/nvcr/nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

WORKDIR /workspace

# Install minimal dependencies for Miniconda download
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (pinned version for reproducibility)
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py312_24.9.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean --all -f -y && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Add conda to PATH (initial setup)
ENV PATH="/opt/conda/bin:$PATH"

# Set LD_LIBRARY_PATH early to help with CUDA detection
ENV LD_LIBRARY_PATH="/opt/conda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Setup conda channels (no update needed - fresh install)
RUN conda config --add channels conda-forge && \
    conda config --add channels pytorch && \
    conda config --add channels nvidia

# Install mamba for faster package resolution
RUN conda install -c conda-forge mamba -y

# Install conda packages with CUDA support using mamba
RUN mamba install -c nvidia -c conda-forge \
    biopython numpy scipy matplotlib pandas \
    pyyaml networkx -y

RUN pip3 install torch torchvision torchaudio

# Install all Python packages with pip (except PyG extensions)
RUN pip install --no-cache-dir \
    torch-geometric \
    e3nn \
    einops \
    ml-collections \
    omegaconf \
    pytorch-lightning \
    deepspeed \
    biopandas \
    tmtools \
    spyrmsd \
    ipywidgets \
    plotly \
    tqdm \
    wandb \
    protobuf \
    pydantic \
    psutil

# Install PyTorch Geometric extensions (assuming CUDA will be available at runtime)
RUN pip install --no-cache-dir --no-index \
    pyg-lib torch-scatter torch-sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.7.0+cu126.html && \
    echo "PyTorch Geometric extensions installed successfully" && \
    python -c "import pyg_lib; print('pyg-lib package imported successfully')" && \
    python -c "import torch_scatter; print('torch-scatter package imported successfully')" && \
    python -c "import torch_sparse; print('torch-sparse package imported successfully')" && \
    python -c "import torch_spline_conv; print('torch-spline-conv package imported successfully')" && \
    python -c "import torch_geometric; print('torch-geometric package imported successfully')" && \
    echo "All PyTorch Geometric components verified successfully"

# Copy YAML configuration files (these change less frequently)
COPY *.yaml *.yml /workspace/

# Copy Python files LAST (these change most frequently)
COPY gvp/ /workspace/gvp/
COPY features/*.py /workspace/features/
COPY flow/*.py /workspace/flow/
COPY data/*.py /workspace/data/
COPY eval/*.py /workspace/eval/
COPY models/*.py /workspace/models/
COPY training/*.py /workspace/training/
COPY df_combined_for_one_hot.csv /workspace/df_combined_for_one_hot.csv


CMD ["/bin/bash"]
