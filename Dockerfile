FROM ubuntu:20.04


RUN apt update && \
    apt upgrade -y && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install -y python3.10 \
    ninja-build

#Building cmake
WORKDIR /opt

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y build-essential libssl-dev wget &&\
    wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz &&\
    tar -zxvf cmake-3.20.0.tar.gz && \
    cd cmake-3.20.0 && \
    ./bootstrap && \
    make && \
    make install

RUN apt install -y python3-pip
RUN apt install -y git
# RUN apt-get update && \
#     apt-get install 

#Building mlir

RUN cd /opt && git clone https://github.com/llvm/llvm-project.git
RUN cd /opt/llvm-project && git reset --hard 26ee8947702d79ce2cab8e577f713685a5ca4a55
RUN mkdir /opt/llvm-project/build
WORKDIR /opt/llvm-project/build

RUN python3 -m pip install numpy pybind11

RUN cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir" \
   -DLLVM_ENABLE_LLD=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD=host \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DPython3_EXECUTABLE=$(which python3) && \
   cmake --build . -j$(nproc) && \
   cmake --build . --target install




