set(LLVM_LINK_COMPONENTS
	Support 
	)


include_directories("include/")
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(include)

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCombineIncGen)

add_llvm_executable(toyc-code
			toyc.cpp
			parser/AST.cpp
			mlir/MLIRGen.cpp
			mlir/Dialect.cpp
			mlir/ToyCombine.cpp
			mlir/ShapeInferencePass.cpp
			mlir/LowerToAffineLoops.cpp
		DEPENDS
		ToyOpsIncGen
		ToyCombineIncGen
		ToyShapeInferenceInterfaceIncGen
		)




target_link_libraries(toyc-code
	PRIVATE
		MLIRAnalysis
		MLIRCastInterfaces
		MLIRCallInterfaces
		MLIRIR
		MLIRParser
		MLIRSideEffectInterfaces
		MLIRTransforms
		MLIRPass)

