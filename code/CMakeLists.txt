set(LLVM_LINK_COMPONENTS
	Support 
	)

add_subdirectory(include)

add_llvm_executable(toyc-code
			toyc.cpp
			parser/AST.cpp
			mlir/MLIRGen.cpp
			mlir/Dialect.cpp
		DEPENDS
		ToyOpsIncGen
		)

include_directories("include/")
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)


target_link_libraries(toyc-code
	PRIVATE
		MLIRAnalysis
		MLIRIR
		MLIRParser
		MLIRSideEffectInterfaces
		MLIRTransforms)

