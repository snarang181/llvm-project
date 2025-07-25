//===- ValueMapper.h - Remapping for constants and metadata -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MapValue interface which is used by various parts of
// the Transforms/Utils library to implement cloning and linking facilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_VALUEMAPPER_H
#define LLVM_TRANSFORMS_UTILS_VALUEMAPPER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/simple_ilist.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class Constant;
class DIBuilder;
class DbgRecord;
class Function;
class GlobalVariable;
class Instruction;
class MDNode;
class Metadata;
class Module;
class Type;
class Value;

using ValueToValueMapTy = ValueMap<const Value *, WeakTrackingVH>;
using DbgRecordIterator = simple_ilist<DbgRecord>::iterator;
using MetadataSetTy = SmallPtrSet<const Metadata *, 16>;
using MetadataPredicate = std::function<bool(const Metadata *)>;

/// This is a class that can be implemented by clients to remap types when
/// cloning constants and instructions.
class LLVM_ABI ValueMapTypeRemapper {
  virtual void anchor(); // Out of line method.

public:
  virtual ~ValueMapTypeRemapper() = default;

  /// The client should implement this method if they want to remap types while
  /// mapping values.
  virtual Type *remapType(Type *SrcTy) = 0;
};

/// This is a class that can be implemented by clients to materialize Values on
/// demand.
class LLVM_ABI ValueMaterializer {
  virtual void anchor(); // Out of line method.

protected:
  ValueMaterializer() = default;
  ValueMaterializer(const ValueMaterializer &) = default;
  ValueMaterializer &operator=(const ValueMaterializer &) = default;
  ~ValueMaterializer() = default;

public:
  /// This method can be implemented to generate a mapped Value on demand. For
  /// example, if linking lazily. Returns null if the value is not materialized.
  virtual Value *materialize(Value *V) = 0;
};

/// These are flags that the value mapping APIs allow.
enum RemapFlags {
  RF_None = 0,

  /// If this flag is set, the remapper knows that only local values within a
  /// function (such as an instruction or argument) are mapped, not global
  /// values like functions and global metadata.
  RF_NoModuleLevelChanges = 1,

  /// If this flag is set, the remapper ignores missing function-local entries
  /// (Argument, Instruction, BasicBlock) that are not in the value map.  If it
  /// is unset, it aborts if an operand is asked to be remapped which doesn't
  /// exist in the mapping.
  ///
  /// There are no such assertions in MapValue(), whose results are almost
  /// unchanged by this flag.  This flag mainly changes the assertion behaviour
  /// in RemapInstruction().
  ///
  /// Since an Instruction's metadata operands (even that point to SSA values)
  /// aren't guaranteed to be dominated by their definitions, MapMetadata will
  /// return "!{}" instead of "null" for \a LocalAsMetadata instances whose SSA
  /// values are unmapped when this flag is set.  Otherwise, \a MapValue()
  /// completely ignores this flag.
  ///
  /// \a MapMetadata() always ignores this flag.
  RF_IgnoreMissingLocals = 2,

  /// Instruct the remapper to reuse and mutate distinct metadata (remapping
  /// them in place) instead of cloning remapped copies. This flag has no
  /// effect when RF_NoModuleLevelChanges, since that implies an identity
  /// mapping.
  RF_ReuseAndMutateDistinctMDs = 4,

  /// Any global values not in value map are mapped to null instead of mapping
  /// to self.  Illegal if RF_IgnoreMissingLocals is also set.
  RF_NullMapMissingGlobalValues = 8,

  /// Do not remap source location atoms. Only safe if to do this if the cloned
  /// instructions being remapped are inserted into a new function, or an
  /// existing function where the inlined-at fields are updated. If in doubt,
  /// don't use this flag. It's used when remapping is known to be un-necessary
  /// to save some compile-time.
  RF_DoNotRemapAtoms = 16,
};

inline RemapFlags operator|(RemapFlags LHS, RemapFlags RHS) {
  return RemapFlags(unsigned(LHS) | unsigned(RHS));
}

/// Context for (re-)mapping values (and metadata).
///
/// A shared context used for mapping and remapping of Value and Metadata
/// instances using \a ValueToValueMapTy, \a RemapFlags, \a
/// ValueMapTypeRemapper, \a ValueMaterializer, and \a IdentityMD.
///
/// There are a number of top-level entry points:
/// - \a mapValue() (and \a mapConstant());
/// - \a mapMetadata() (and \a mapMDNode());
/// - \a remapInstruction();
/// - \a remapFunction(); and
/// - \a remapGlobalObjectMetadata().
///
/// The \a ValueMaterializer can be used as a callback, but cannot invoke any
/// of these top-level functions recursively.  Instead, callbacks should use
/// one of the following to schedule work lazily in the \a ValueMapper
/// instance:
/// - \a scheduleMapGlobalInitializer()
/// - \a scheduleMapAppendingVariable()
/// - \a scheduleMapGlobalAlias()
/// - \a scheduleMapGlobalIFunc()
/// - \a scheduleRemapFunction()
///
/// Sometimes a callback needs a different mapping context.  Such a context can
/// be registered using \a registerAlternateMappingContext(), which takes an
/// alternate \a ValueToValueMapTy and \a ValueMaterializer and returns a ID to
/// pass into the schedule*() functions.
///
/// If an \a IdentityMD predicate is optionally provided, \a Metadata for which
/// the predicate returns true will be mapped onto itself in \a VM on first use.
///
/// TODO: lib/Linker really doesn't need the \a ValueHandle in the \a
/// ValueToValueMapTy.  We should template \a ValueMapper (and its
/// implementation classes), and explicitly instantiate on two concrete
/// instances of \a ValueMap (one as \a ValueToValueMap, and one with raw \a
/// Value pointers).  It may be viable to do away with \a TrackingMDRef in the
/// \a Metadata side map for the lib/Linker case as well, in which case we'll
/// need a new template parameter on \a ValueMap.
///
/// TODO: Update callers of \a RemapInstruction() and \a MapValue() (etc.) to
/// use \a ValueMapper directly.
class ValueMapper {
  void *pImpl;

public:
  LLVM_ABI ValueMapper(ValueToValueMapTy &VM, RemapFlags Flags = RF_None,
                       ValueMapTypeRemapper *TypeMapper = nullptr,
                       ValueMaterializer *Materializer = nullptr,
                       const MetadataPredicate *IdentityMD = nullptr);
  ValueMapper(ValueMapper &&) = delete;
  ValueMapper(const ValueMapper &) = delete;
  ValueMapper &operator=(ValueMapper &&) = delete;
  ValueMapper &operator=(const ValueMapper &) = delete;
  LLVM_ABI ~ValueMapper();

  /// Register an alternate mapping context.
  ///
  /// Returns a MappingContextID that can be used with the various schedule*()
  /// API to switch in a different value map on-the-fly.
  LLVM_ABI unsigned
  registerAlternateMappingContext(ValueToValueMapTy &VM,
                                  ValueMaterializer *Materializer = nullptr);

  /// Add to the current \a RemapFlags.
  ///
  /// \note Like the top-level mapping functions, \a addFlags() must be called
  /// at the top level, not during a callback in a \a ValueMaterializer.
  LLVM_ABI void addFlags(RemapFlags Flags);

  LLVM_ABI Metadata *mapMetadata(const Metadata &MD);
  LLVM_ABI MDNode *mapMDNode(const MDNode &N);

  LLVM_ABI Value *mapValue(const Value &V);
  LLVM_ABI Constant *mapConstant(const Constant &C);

  LLVM_ABI void remapInstruction(Instruction &I);
  LLVM_ABI void remapDbgRecord(Module *M, DbgRecord &V);
  LLVM_ABI void remapDbgRecordRange(Module *M,
                                    iterator_range<DbgRecordIterator> Range);
  LLVM_ABI void remapFunction(Function &F);
  LLVM_ABI void remapGlobalObjectMetadata(GlobalObject &GO);

  LLVM_ABI void scheduleMapGlobalInitializer(GlobalVariable &GV, Constant &Init,
                                             unsigned MappingContextID = 0);
  LLVM_ABI void scheduleMapAppendingVariable(GlobalVariable &GV,
                                             Constant *InitPrefix,
                                             bool IsOldCtorDtor,
                                             ArrayRef<Constant *> NewMembers,
                                             unsigned MappingContextID = 0);
  LLVM_ABI void scheduleMapGlobalAlias(GlobalAlias &GA, Constant &Aliasee,
                                       unsigned MappingContextID = 0);
  LLVM_ABI void scheduleMapGlobalIFunc(GlobalIFunc &GI, Constant &Resolver,
                                       unsigned MappingContextID = 0);
  LLVM_ABI void scheduleRemapFunction(Function &F,
                                      unsigned MappingContextID = 0);
};

/// Look up or compute a value in the value map.
///
/// Return a mapped value for a function-local value (Argument, Instruction,
/// BasicBlock), or compute and memoize a value for a Constant.
///
///  1. If \c V is in VM, return the result.
///  2. Else if \c V can be materialized with \c Materializer, do so, memoize
///     it in \c VM, and return it.
///  3. Else if \c V is a function-local value, return nullptr.
///  4. Else if \c V is a \a GlobalValue, return \c nullptr or \c V depending
///     on \a RF_NullMapMissingGlobalValues.
///  5. Else if \c V is a \a MetadataAsValue wrapping a LocalAsMetadata,
///     recurse on the local SSA value, and return nullptr or "metadata !{}" on
///     missing depending on RF_IgnoreMissingValues.
///  6. Else if \c V is a \a MetadataAsValue, rewrap the return of \a
///     MapMetadata().
///  7. Else, compute the equivalent constant, and return it.
inline Value *MapValue(const Value *V, ValueToValueMapTy &VM,
                       RemapFlags Flags = RF_None,
                       ValueMapTypeRemapper *TypeMapper = nullptr,
                       ValueMaterializer *Materializer = nullptr,
                       const MetadataPredicate *IdentityMD = nullptr) {
  return ValueMapper(VM, Flags, TypeMapper, Materializer, IdentityMD)
      .mapValue(*V);
}

/// Lookup or compute a mapping for a piece of metadata.
///
/// Compute and memoize a mapping for \c MD.
///
///  1. If \c MD is mapped, return it.
///  2. Else if \a RF_NoModuleLevelChanges or \c MD is an \a MDString, return
///     \c MD.
///  3. Else if \c MD is a \a ConstantAsMetadata, call \a MapValue() and
///     re-wrap its return (returning nullptr on nullptr).
///  4. Else if \c IdentityMD predicate returns true for \c MD then add an
///     identity mapping for it and return it.
///  5. Else, \c MD is an \a MDNode.  These are remapped, along with their
///     transitive operands.  Distinct nodes are duplicated or moved depending
///     on \a RF_MoveDistinctNodes.  Uniqued nodes are remapped like constants.
///
/// \note \a LocalAsMetadata is completely unsupported by \a MapMetadata.
/// Instead, use \a MapValue() with its wrapping \a MetadataAsValue instance.
inline Metadata *MapMetadata(const Metadata *MD, ValueToValueMapTy &VM,
                             RemapFlags Flags = RF_None,
                             ValueMapTypeRemapper *TypeMapper = nullptr,
                             ValueMaterializer *Materializer = nullptr,
                             const MetadataPredicate *IdentityMD = nullptr) {
  return ValueMapper(VM, Flags, TypeMapper, Materializer, IdentityMD)
      .mapMetadata(*MD);
}

/// Version of MapMetadata with type safety for MDNode.
inline MDNode *MapMetadata(const MDNode *MD, ValueToValueMapTy &VM,
                           RemapFlags Flags = RF_None,
                           ValueMapTypeRemapper *TypeMapper = nullptr,
                           ValueMaterializer *Materializer = nullptr,
                           const MetadataPredicate *IdentityMD = nullptr) {
  return ValueMapper(VM, Flags, TypeMapper, Materializer, IdentityMD)
      .mapMDNode(*MD);
}

/// Convert the instruction operands from referencing the current values into
/// those specified by VM.
///
/// If \a RF_IgnoreMissingLocals is set and an operand can't be found via \a
/// MapValue(), use the old value.  Otherwise assert that this doesn't happen.
///
/// Note that \a MapValue() only returns \c nullptr for SSA values missing from
/// \c VM.
inline void RemapInstruction(Instruction *I, ValueToValueMapTy &VM,
                             RemapFlags Flags = RF_None,
                             ValueMapTypeRemapper *TypeMapper = nullptr,
                             ValueMaterializer *Materializer = nullptr,
                             const MetadataPredicate *IdentityMD = nullptr) {
  ValueMapper(VM, Flags, TypeMapper, Materializer, IdentityMD)
      .remapInstruction(*I);
}

/// Remap source location atom. Called by RemapInstruction. This updates the
/// instruction's atom group number if it has been mapped (e.g. with
/// llvm::mapAtomInstance), which is necessary to distinguish source code
/// atoms on duplicated code paths.
LLVM_ABI void RemapSourceAtom(Instruction *I, ValueToValueMapTy &VM);

/// Remap the Values used in the DbgRecord \a DR using the value map \a
/// VM.
inline void RemapDbgRecord(Module *M, DbgRecord *DR, ValueToValueMapTy &VM,
                           RemapFlags Flags = RF_None,
                           ValueMapTypeRemapper *TypeMapper = nullptr,
                           ValueMaterializer *Materializer = nullptr,
                           const MetadataPredicate *IdentityMD = nullptr) {
  ValueMapper(VM, Flags, TypeMapper, Materializer, IdentityMD)
      .remapDbgRecord(M, *DR);
}

/// Remap the Values used in the DbgRecords \a Range using the value map \a
/// VM.
inline void RemapDbgRecordRange(Module *M,
                                iterator_range<DbgRecordIterator> Range,
                                ValueToValueMapTy &VM,
                                RemapFlags Flags = RF_None,
                                ValueMapTypeRemapper *TypeMapper = nullptr,
                                ValueMaterializer *Materializer = nullptr,
                                const MetadataPredicate *IdentityMD = nullptr) {
  ValueMapper(VM, Flags, TypeMapper, Materializer, IdentityMD)
      .remapDbgRecordRange(M, Range);
}

/// Remap the operands, metadata, arguments, and instructions of a function.
///
/// Calls \a MapValue() on prefix data, prologue data, and personality
/// function; calls \a MapMetadata() on each attached MDNode; remaps the
/// argument types using the provided \c TypeMapper; and calls \a
/// RemapInstruction() on every instruction.
inline void RemapFunction(Function &F, ValueToValueMapTy &VM,
                          RemapFlags Flags = RF_None,
                          ValueMapTypeRemapper *TypeMapper = nullptr,
                          ValueMaterializer *Materializer = nullptr,
                          const MetadataPredicate *IdentityMD = nullptr) {
  ValueMapper(VM, Flags, TypeMapper, Materializer, IdentityMD).remapFunction(F);
}

/// Version of MapValue with type safety for Constant.
inline Constant *MapValue(const Constant *V, ValueToValueMapTy &VM,
                          RemapFlags Flags = RF_None,
                          ValueMapTypeRemapper *TypeMapper = nullptr,
                          ValueMaterializer *Materializer = nullptr,
                          const MetadataPredicate *IdentityMD = nullptr) {
  return ValueMapper(VM, Flags, TypeMapper, Materializer, IdentityMD)
      .mapConstant(*V);
}

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_VALUEMAPPER_H
