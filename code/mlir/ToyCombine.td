
#ifndef TOY_COMBINE
#define TOY_COMBINE

include "mlir/IR/PatternBase.td"
include "toy/Ops.td"

def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
								(ReshapeOp $arg)>;


def ReshapeConstant : 
	NativeCodeCall<"$0.reshape(::llvm::cast<ShapedType>($1.getType()))">;

def FoldConstantReshapeOptPattern : Pat<
	(ReshapeOp:$res (ConstantOp $arg)),
	(ConstantOp (ReshapeConstant $arg, $res))>;


def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;

def RedundantReshapeOptPattern : Pat<
	(ReshapeOp:$res $arg), (replaceWithValue $arg),
	[(TypesAreIdentical $res, $arg)]>;

#endif  //TOY_COMBINE