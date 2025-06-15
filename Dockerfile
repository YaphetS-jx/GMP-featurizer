# Use a minimal base image with build tools
FROM gcc:latest

# Set maintainer label
LABEL maintainer="jingxin.bc@gmail.com"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install only necessary build tools and dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    # Build essentials (gcc is already included in base image)
    cmake \
    make \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Boost with retry mechanism
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Create directories for external libraries
RUN mkdir -p /usr/local/src

# Install nlohmann/json (header-only library)
RUN cd /usr/local/src && \
    git clone https://github.com/nlohmann/json.git

# Install GEMMI
RUN cd /usr/local/src && \
    git clone https://github.com/project-gemmi/gemmi.git

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
