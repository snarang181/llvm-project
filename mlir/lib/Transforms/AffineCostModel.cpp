
#include "mlir/Transforms/AffineCostModel.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
struct AffineCostModelPass
    : public PassWrapper<AffineCostModelPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    auto func = getOperation();

    func->walk([&](affine::AffineForOp forOp) {
      llvm::errs() << "AffineForOp found at ";
      forOp.getLoc().print(llvm::errs());
      llvm::errs() << "\n";
    });
  }

  StringRef getArgument() const override { return "affine-cost-model"; }
};
} // namespace mlir

std::unique_ptr<mlir::Pass> mlir::createAffineCostModelPass() {
  return std::make_unique<AffineCostModelPass>();
}