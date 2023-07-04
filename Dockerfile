FROM ubuntu:20.04


RUN apt update && \
    apt install -y python \
    cmake 

RUN apt install -y python3-pip
RUN apt install -y git
# RUN apt-get update && \
#     apt-get install 

#RUN mkdir /opt
COPY requirements.txt /opt/requirements.txt
COPY build-requirements.txt /opt/build-requirements.txt
COPY pytorch-requirements.txt /opt/pytorch-requirements.txt
COPY test-requirements.txt /opt/test-requirements.txt
COPY torchvision-requirements.txt /opt/torchvision-requirements.txt

RUN python3 -m pip install -r /opt/requirements.txt

#Building mlir

RUN cd /opt && git clone https://github.com/llvm/llvm-project.git
RUN cd /opt/llvm-project && git reset --hard 26ee8947702d79ce2cab8e577f713685a5ca4a55
RUN mkdir /opt/llvm-project/build
WORKDIR /opt/llvm-project/build

RUN apt-get update && \
    apt-get install -y clang lld lldb ninja-build

RUN cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir" \
   -DLLVM_ENABLE_LLD=ON \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DCMAKE_LINKER=lld \
   -DLLVM_INSTALL_UTILS=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD=host \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DPython3_EXECUTABLE=$(which python3) && \
   cmake --build . -j$(nproc) && \
   cmake --build . --target install




