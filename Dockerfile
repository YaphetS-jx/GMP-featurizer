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
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Create a user with the same UID as the host user
ARG USER_ID=10101
ARG GROUP_ID=10101
RUN groupadd -f -g $GROUP_ID appuser && \
    id -u appuser || useradd -u $USER_ID -g $GROUP_ID -m -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Create directories for external libraries
RUN mkdir -p /usr/local/src

# Install GEMMI
RUN cd /usr/local/src && \
    git clone https://github.com/project-gemmi/gemmi.git

# Install nlohmann/json
RUN cd /usr/local/src && \
    git clone --depth 1 --branch v3.11.3 https://github.com/nlohmann/json.git
    
# Configure readline for better command-line experience
RUN echo '"\e[A": history-search-backward' >> /home/appuser/.inputrc && \
    echo '"\e[B": history-search-forward' >> /home/appuser/.inputrc && \
    echo 'set completion-ignore-case on' >> /home/appuser/.inputrc && \
    echo 'set show-all-if-ambiguous on' >> /home/appuser/.inputrc && \
    echo 'set show-all-if-unmodified on' >> /home/appuser/.inputrc

# Copy your application files
COPY . .

# Change ownership of the app directory
RUN chown -R appuser:appuser /app

# Switch to the appuser
USER appuser

# Set default command
CMD ["/bin/bash"]
