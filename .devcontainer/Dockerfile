FROM mcr.microsoft.com/devcontainers/cpp:0-ubuntu-22.04

RUN apt-get update \
        && export DEBIAN_FRONTEND=noninteractive \
        && apt-get -y install --no-install-recommends clang-format openmpi-bin libopenmpi-dev \
        && apt-get autoremove -y \
        && apt-get clean -y \
        && rm -rf /var/lib/apt/lists/*

# Download and Install Catch2
RUN     mkdir /usr/local/src/Catch2 \
        && cd /usr/local/src/Catch2 \
        && git clone https://github.com/catchorg/Catch2.git \
        && cd Catch2 \
        && cmake -B build -DBUILD_TESTING=OFF \
        && cmake --build build \
        && cmake --install build/

ENV LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH