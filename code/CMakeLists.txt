set(LLVM_LINK_COMPONENTS
	Support 
	)

add_subdirectory(include)

include_directories("include/")
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})


set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCombineIncGen)

add_llvm_executable(toyc-code
			toyc.cpp
			parser/AST.cpp
			mlir/MLIRGen.cpp
			mlir/Dialect.cpp
			mlir/ToyCombine.cpp
		DEPENDS
		ToyOpsIncGen
		ToyCombineIncGen
		)




target_link_libraries(toyc-code
	PRIVATE
		MLIRAnalysis
		MLIRIR
		MLIRParser
		MLIRSideEffectInterfaces
		MLIRTransforms
		MLIRPass)

