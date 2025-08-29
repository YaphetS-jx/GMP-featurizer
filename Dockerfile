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
  && update-ca-certificates \
  && rm -rf /var/lib/apt/lists/*

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

# Set default command
CMD ["/bin/bash"]
