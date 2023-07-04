cmake_minimum_required(VERSION 3.20.0)

project(MLIR-Toy)
find_package(LLVM)
find_package(MLIR)

include(AddLLVM)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

add_custom_target(Toy)
set_target_properties(Toy PROPERTIES FOLDER Examples)

macro(add_toy_chapter name)
  add_dependencies(Toy ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_toy_chapter name)

add_subdirectory(code)