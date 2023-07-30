#include "mlir/Pass/Pass.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "toy/ShapeInferenceInterface.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

#define DEBUG_TYPE "shape-inference"

using namespace mlir;
using namespace toy;

#include "toy/ShapeInferenceOpInterfaces.cpp.inc"

namespace {


	struct ShapeInferencePass
		: public mlir::PassWrapper<ShapeInferencePass, OperationPass<toy::FuncOp>>
	{
		MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)

		void runOnOperation() override {

			auto f = getOperation();

			llvm::SmallPtrSet<mlir::Operation*, 16> opWorklist;
			f.walk([&](mlir::Operation* op)
				{
					if (returnsDynamicShape(op))
					opWorklist.insert(op);
				});

			while (!opWorklist.empty())
			{
				auto nextop = llvm::find_if(opWorklist, allOperandsInferred);
				if (nextop == opWorklist.end())
					break;

				Operation* op = *nextop;
				opWorklist.erase(op);

				if (auto shapeOp = dyn_cast<ShapeInference>(op))
				{
					shapeOp.inferShapes();
				}
				else {
					op->emitError("unable to infer shape of operation without shape "
						"inference interface");
					return signalPassFailure();
				}

			}

			if (!opWorklist.empty())
			{
				f.emitError("Shape inference failed, ")
					<< opWorklist.size() << " operation couldn't be inferred\n";

				signalPassFailure();
			}
		}

		static bool allOperandsInferred(Operation* op)
		{
			return llvm::all_of(op->getOperandTypes(), [](Type operandType) {
				return llvm::isa<RankedTensorType>(operandType);
				});
		}

		static bool returnsDynamicShape(Operation* op) {
			return llvm::any_of(op->getResultTypes(), [](Type resultType) {
				return !llvm::isa<RankedTensorType>(resultType);
				});
		}
	};
}

std::unique_ptr<mlir::Pass> mlir::toy::createShapeInferencePass()
{
	return std::make_unique<ShapeInferencePass>();
}