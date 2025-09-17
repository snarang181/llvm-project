
#include "mlir/Transforms/AffineCostModel.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
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

      // If there is a trip count, let's get that.
      // getConstantTripCount takes a forOp as its argument.
      auto tripCount = getConstantTripCount(forOp);
      if (!tripCount.has_value())
        llvm::errs() << "  Trip count: unknown\n";
      else
        llvm::errs() << "  Trip count: " << tripCount.value() << "\n";

      // Count memory ops, i.e. loads and stores.
      unsigned numLoads = 0, numStores = 0;
      forOp->walk([&](Operation *op) {
        if (isa<affine::AffineLoadOp>(op))
          ++numLoads;
        else if (isa<affine::AffineStoreOp>(op))
          ++numStores;
      });
      llvm::errs() << "  Memory ops: " << (numLoads + numStores)
                   << " (loads: " << numLoads << ", stores: " << numStores
                   << ")\n";
    });
  }

  StringRef getArgument() const override { return "affine-cost-model"; }
};
} // namespace mlir

std::unique_ptr<mlir::Pass> mlir::createAffineCostModelPass() {
  return std::make_unique<AffineCostModelPass>();
}