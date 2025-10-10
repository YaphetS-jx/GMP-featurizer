# Use Ubuntu 24.04 as base image
FROM ubuntu:24.04

# Set maintainer label
LABEL maintainer="jingxin.bc@gmail.com"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools and additional tools
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    vim \
    cmake \
    curl \
    ca-certificates \
    tar \
    gdb gdbserver \
    # Python and pybind11 dependencies
    python3 \
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-matplotlib \
    python3-setuptools \
    python3-wheel \
    pkg-config \
  && update-ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install pybind11 via pip
RUN pip3 install --break-system-packages pybind11[global]>=2.6.0

# Set working directory
WORKDIR /app

# Create directories for external libraries
RUN mkdir -p /usr/local/src

# Install GEMMI and JSON
RUN cd /usr/local/src && \
    git clone https://github.com/project-gemmi/gemmi.git && \
    git clone --depth 1 --branch v3.11.3 https://github.com/nlohmann/json.git
    
# Configure readline for better command-line experience
RUN echo '"\e[A": history-search-backward' >> /root/.inputrc && \
    echo '"\e[B": history-search-forward' >> /root/.inputrc && \
    echo 'set completion-ignore-case on' >> /root/.inputrc && \
    echo 'set show-all-if-ambiguous on' >> /root/.inputrc && \
    echo 'set show-all-if-unmodified on' >> /root/.inputrc

# Copy your application files
COPY . .

# Create a symlink for python command
RUN ln -s /usr/bin/python3 /usr/bin/python

# Example: Build the Python module (uncomment to build automatically)
# RUN mkdir -p build && cd build && \
#     cmake -DBUILD_PYTHON_MODULE=ON .. && \
#     make gmp_featurizer

# Set default command
CMD ["/bin/bash"]
