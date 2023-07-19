#ifndef TOY_MLIRGEN_H
#define TOY_MLIRGEN_H

#include <memory>


namespace mlir {
	class MLIRContext;
	template <typename OpTy>
	class OwningOpref;
	class ModuleOp;
}


namespace toy {
	class ModuleAST;

	mlir::OwningOpref<mlir::ModuleOp> mlirGen(mlir::MLIRContext& context,
		ModuleAST& moduleAST);
}

#endif