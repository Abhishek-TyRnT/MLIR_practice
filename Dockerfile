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


#Install bazel
RUN apt install apt-transport-https curl gnupg -y && \
    curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel-archive-keyring.gpg && \
    mv bazel-archive-keyring.gpg /usr/share/keyrings && \
    echo "deb [signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
    apt update && \
    apt install bazel-6.1.0 clang -y






