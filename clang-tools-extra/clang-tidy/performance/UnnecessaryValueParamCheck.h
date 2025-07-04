//===--- UnnecessaryValueParamCheck.h - clang-tidy---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_UNNECESSARY_VALUE_PARAM_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_UNNECESSARY_VALUE_PARAM_H

#include "../ClangTidyCheck.h"
#include "../utils/IncludeInserter.h"
#include "clang/Analysis/Analyses/ExprMutationAnalyzer.h"

namespace clang::tidy::performance {

/// A check that flags value parameters of expensive to copy types that
/// can safely be converted to const references.
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/performance/unnecessary-value-param.html
class UnnecessaryValueParamCheck : public ClangTidyCheck {
public:
  UnnecessaryValueParamCheck(StringRef Name, ClangTidyContext *Context);
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override {
    return LangOpts.CPlusPlus;
  }
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void storeOptions(ClangTidyOptions::OptionMap &Opts) override;
  void onEndOfTranslationUnit() override;

protected:
  // Create diagnostics. These are virtual so that derived classes can change
  // behaviour.
  virtual void handleMoveFix(const ParmVarDecl &Param,
                             const DeclRefExpr &CopyArgument,
                             ASTContext &Context);
  virtual void handleConstRefFix(const FunctionDecl &Function,
                                 const ParmVarDecl &Param, ASTContext &Context);

private:
  ExprMutationAnalyzer::Memoized MutationAnalyzerCache;
  utils::IncludeInserter Inserter;
  const std::vector<StringRef> AllowedTypes;
  bool IgnoreCoroutines;
};

} // namespace clang::tidy::performance

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_PERFORMANCE_UNNECESSARY_VALUE_PARAM_H
