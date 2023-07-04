set(LLVM_LINK_COMPONENTS
	Support 
	)

add_toy_chapter(toyc-code
			toyc.cpp
			parser/AST.cpp
		)

include_directories(include/)

target_link_libraries(toyc-code
	PRIVATE
		MLIRSupport)