//===- TestTransformOps.td ---------------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_TRANSFORM_OPS
#define TEST_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

/// Transform dialect operations for testing transformations in MLIR

def TestMoveOperandDeps :
    Op<Transform_Dialect, "test.move_operand_deps",
        [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
         DeclareOpInterfaceMethods<TransformOpInterface>,
         ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Moves all dependencies of on operation before another operation.
  }];

  let arguments =
    (ins TransformHandleTypeInterface:$op,
         TransformHandleTypeInterface:$insertion_point);
  
  let results = (outs);

  let assemblyFormat = [{
    $op `before` $insertion_point attr-dict
    `:` type($op) `,` type($insertion_point)
  }];
}

def TestMoveValueDefns :
    Op<Transform_Dialect, "test.move_value_defns",
        [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
         DeclareOpInterfaceMethods<TransformOpInterface>,
         ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Moves all dependencies of on operation before another operation.
  }];

  let arguments =
    (ins Variadic<TransformValueHandleTypeInterface>:$values,
         TransformHandleTypeInterface:$insertion_point);
  
  let results = (outs);

  let assemblyFormat = [{
    $values `before` $insertion_point attr-dict
    `:` `(` type($values) `)` `` `,` type($insertion_point)
  }];
}

//===----------------------------------------------------------------------===//
// Test affine functionality.
//===----------------------------------------------------------------------===//

def TestMakeComposedFoldedAffineApply :
    Op<Transform_Dialect, "test.make_composed_folded_affine_apply",
        [FunctionalStyleTransformOpTrait, 
         MemoryEffectsOpInterface,
         TransformOpInterface,
         TransformEachOpTrait,
         ReportTrackingListenerFailuresOpTrait]> {
  let description = [{
    Rewrite an affine_apply by using the makeComposedFoldedAffineApply API.
  }];
  let arguments = (ins TransformHandleTypeInterface:$op);
  let results = (outs TransformHandleTypeInterface:$composed);
  let assemblyFormat = [{
    $op attr-dict `:` functional-type(operands, results)
  }];
  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::affine::AffineApplyOp affineApplyOp,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // TEST_TRANSFORM_OPS
