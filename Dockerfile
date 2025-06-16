# Use Ubuntu 24.04 as base image
FROM ubuntu:24.04

# Set maintainer label
LABEL maintainer="jingxin.bc@gmail.com"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools and dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    cmake \
    make \
    # Boost libraries
    libboost-all-dev \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create directories for external libraries
RUN mkdir -p /usr/local/src

# Install GEMMI
RUN cd /usr/local/src && \
    git clone https://github.com/project-gemmi/gemmi.git

# Install xsimd
RUN cd /usr/local/src && \
    git clone https://github.com/xtensor-stack/xsimd.git
    
# Configure readline for better command-line experience
RUN echo '"\e[A": history-search-backward' >> /root/.inputrc && \
    echo '"\e[B": history-search-forward' >> /root/.inputrc && \
    echo 'set completion-ignore-case on' >> /root/.inputrc && \
    echo 'set show-all-if-ambiguous on' >> /root/.inputrc && \
    echo 'set show-all-if-unmodified on' >> /root/.inputrc

# Copy your application files
COPY . .

# Set default command
CMD ["/bin/bash"]
