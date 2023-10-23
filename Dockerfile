FROM ubuntu:22.04    

#Building cmake
WORKDIR /opt

RUN apt-get update && \
    apt-get install -y build-essential

RUN apt-get install -y wget git libssl-dev &&\
    wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0.tar.gz &&\
    tar -zxvf cmake-3.20.0.tar.gz && \
    cd cmake-3.20.0 && \
    ./bootstrap && \
    make && \
    make install


#Building mlir

RUN cd /opt && \
    wget https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.3/llvm-project-17.0.3.src.tar.xz && \
    tar -xf llvm-project-17.0.3.src.tar.xz && \
    cd llvm-project-17.0.3.src && \
    mkdir build

WORKDIR /opt/llvm-project-17.0.3.src/build

RUN apt-get install -y python3 python3-pip ninja-build clang && \
    python3 -m pip install numpy pybind11

RUN cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;lld" \
   -DLLVM_INSTALL_UTILS=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD=host \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DCMAKE_C_COMPILER=clang \
   -DPython3_EXECUTABLE=$(which python3) && \
   cmake --build . -j$(nproc) && \
   cmake --build . --target install




