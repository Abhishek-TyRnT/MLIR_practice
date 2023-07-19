cmake_minimum_required(VERSION 3.20.0)

project(MLIR-Toy)

set(CMAKE_CXX_STANDARD 17)
find_package(LLVM)
find_package(MLIR)

set(MLIR_CMAKE_DIR "/usr/local/lib/cmake/mlir")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(AddMLIR)
include(AddLLVM)
include(TableGen)

add_custom_target(Toy)
set_target_properties(Toy PROPERTIES FOLDER Examples)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
    "${CMAKE_CURRENT_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")

macro(add_toy_chapter name)
  add_dependencies(Toy ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_toy_chapter name)



add_subdirectory(code)