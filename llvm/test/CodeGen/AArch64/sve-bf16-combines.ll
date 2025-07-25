; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -mattr=+sve,+bf16,+sve-b16b16 < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

define <vscale x 8 x bfloat> @fmla_nxv8bf16(<vscale x 8 x bfloat> %acc, <vscale x 8 x bfloat> %m1, <vscale x 8 x bfloat> %m2) {
; CHECK-LABEL: fmla_nxv8bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h
; CHECK-NEXT:    bfmla z0.h, p0/m, z1.h, z2.h
; CHECK-NEXT:    ret
  %mul = fmul contract <vscale x 8 x bfloat> %m1, %m2
  %res = fadd contract <vscale x 8 x bfloat> %acc, %mul
  ret <vscale x 8 x bfloat> %res
}

define <vscale x 4 x bfloat> @fmla_nxv4bf16(<vscale x 4 x bfloat> %acc, <vscale x 4 x bfloat> %m1, <vscale x 4 x bfloat> %m2) {
; CHECK-LABEL: fmla_nxv4bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s
; CHECK-NEXT:    bfmla z0.h, p0/m, z1.h, z2.h
; CHECK-NEXT:    ret
  %mul = fmul contract <vscale x 4 x bfloat> %m1, %m2
  %res = fadd contract <vscale x 4 x bfloat> %acc, %mul
  ret <vscale x 4 x bfloat> %res
}

define <vscale x 2 x bfloat> @fmla_nxv2bf16(<vscale x 2 x bfloat> %acc, <vscale x 2 x bfloat> %m1, <vscale x 2 x bfloat> %m2) {
; CHECK-LABEL: fmla_nxv2bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d
; CHECK-NEXT:    bfmla z0.h, p0/m, z1.h, z2.h
; CHECK-NEXT:    ret
  %mul = fmul contract <vscale x 2 x bfloat> %m1, %m2
  %res = fadd contract <vscale x 2 x bfloat> %acc, %mul
  ret <vscale x 2 x bfloat> %res
}

define <vscale x 8 x bfloat> @fmls_nxv8bf16(<vscale x 8 x bfloat> %acc, <vscale x 8 x bfloat> %m1, <vscale x 8 x bfloat> %m2) {
; CHECK-LABEL: fmls_nxv8bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.h
; CHECK-NEXT:    bfmls z0.h, p0/m, z1.h, z2.h
; CHECK-NEXT:    ret
  %mul = fmul contract <vscale x 8 x bfloat> %m1, %m2
  %res = fsub contract <vscale x 8 x bfloat> %acc, %mul
  ret <vscale x 8 x bfloat> %res
}

define <vscale x 4 x bfloat> @fmls_nxv4bf16(<vscale x 4 x bfloat> %acc, <vscale x 4 x bfloat> %m1, <vscale x 4 x bfloat> %m2) {
; CHECK-LABEL: fmls_nxv4bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s
; CHECK-NEXT:    bfmls z0.h, p0/m, z1.h, z2.h
; CHECK-NEXT:    ret
  %mul = fmul contract <vscale x 4 x bfloat> %m1, %m2
  %res = fsub contract <vscale x 4 x bfloat> %acc, %mul
  ret <vscale x 4 x bfloat> %res
}

define <vscale x 2 x bfloat> @fmls_nxv2bf16(<vscale x 2 x bfloat> %acc, <vscale x 2 x bfloat> %m1, <vscale x 2 x bfloat> %m2) {
; CHECK-LABEL: fmls_nxv2bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.d
; CHECK-NEXT:    bfmls z0.h, p0/m, z1.h, z2.h
; CHECK-NEXT:    ret
  %mul = fmul contract <vscale x 2 x bfloat> %m1, %m2
  %res = fsub contract <vscale x 2 x bfloat> %acc, %mul
  ret <vscale x 2 x bfloat> %res
}

define <vscale x 8 x bfloat> @fmla_sel_nxv8bf16(<vscale x 8 x i1> %pred, <vscale x 8 x bfloat> %acc, <vscale x 8 x bfloat> %m1, <vscale x 8 x bfloat> %m2) {
; CHECK-LABEL: fmla_sel_nxv8bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    bfmla z0.h, p0/m, z1.h, z2.h
; CHECK-NEXT:    ret
  %mul = fmul contract <vscale x 8 x bfloat> %m1, %m2
  %add = fadd contract <vscale x 8 x bfloat> %acc, %mul
  %res = select <vscale x 8 x i1> %pred, <vscale x 8 x bfloat> %add, <vscale x 8 x bfloat> %acc
  ret <vscale x 8 x bfloat> %res
}

define <vscale x 4 x bfloat> @fmla_sel_nxv4bf16(<vscale x 4 x i1> %pred, <vscale x 4 x bfloat> %acc, <vscale x 4 x bfloat> %m1, <vscale x 4 x bfloat> %m2) {
; CHECK-LABEL: fmla_sel_nxv4bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    bfmla z0.h, p0/m, z1.h, z2.h
; CHECK-NEXT:    ret
  %mul = fmul contract <vscale x 4 x bfloat> %m1, %m2
  %add = fadd contract <vscale x 4 x bfloat> %acc, %mul
  %res = select <vscale x 4 x i1> %pred, <vscale x 4 x bfloat> %add, <vscale x 4 x bfloat> %acc
  ret <vscale x 4 x bfloat> %res
}

define <vscale x 2 x bfloat> @fmla_sel_nxv2bf16(<vscale x 2 x i1> %pred, <vscale x 2 x bfloat> %acc, <vscale x 2 x bfloat> %m1, <vscale x 2 x bfloat> %m2) {
; CHECK-LABEL: fmla_sel_nxv2bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    bfmla z0.h, p0/m, z1.h, z2.h
; CHECK-NEXT:    ret
  %mul = fmul contract <vscale x 2 x bfloat> %m1, %m2
  %add = fadd contract <vscale x 2 x bfloat> %acc, %mul
  %res = select <vscale x 2 x i1> %pred, <vscale x 2 x bfloat> %add, <vscale x 2 x bfloat> %acc
  ret <vscale x 2 x bfloat> %res
}

define <vscale x 8 x bfloat> @fmls_sel_nxv8bf16(<vscale x 8 x i1> %pred, <vscale x 8 x bfloat> %acc, <vscale x 8 x bfloat> %m1, <vscale x 8 x bfloat> %m2) {
; CHECK-LABEL: fmls_sel_nxv8bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    bfmls z0.h, p0/m, z1.h, z2.h
; CHECK-NEXT:    ret
  %mul = fmul contract <vscale x 8 x bfloat> %m1, %m2
  %sub = fsub contract <vscale x 8 x bfloat> %acc, %mul
  %res = select <vscale x 8 x i1> %pred, <vscale x 8 x bfloat> %sub, <vscale x 8 x bfloat> %acc
  ret <vscale x 8 x bfloat> %res
}

define <vscale x 4 x bfloat> @fmls_sel_nxv4bf16(<vscale x 4 x i1> %pred, <vscale x 4 x bfloat> %acc, <vscale x 4 x bfloat> %m1, <vscale x 4 x bfloat> %m2) {
; CHECK-LABEL: fmls_sel_nxv4bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    bfmls z0.h, p0/m, z1.h, z2.h
; CHECK-NEXT:    ret
  %mul = fmul contract <vscale x 4 x bfloat> %m1, %m2
  %sub = fsub contract <vscale x 4 x bfloat> %acc, %mul
  %res = select <vscale x 4 x i1> %pred, <vscale x 4 x bfloat> %sub, <vscale x 4 x bfloat> %acc
  ret <vscale x 4 x bfloat> %res
}

define <vscale x 2 x bfloat> @fmls_sel_nxv2bf16(<vscale x 2 x i1> %pred, <vscale x 2 x bfloat> %acc, <vscale x 2 x bfloat> %m1, <vscale x 2 x bfloat> %m2) {
; CHECK-LABEL: fmls_sel_nxv2bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    bfmls z0.h, p0/m, z1.h, z2.h
; CHECK-NEXT:    ret
  %mul = fmul contract <vscale x 2 x bfloat> %m1, %m2
  %sub = fsub contract <vscale x 2 x bfloat> %acc, %mul
  %res = select <vscale x 2 x i1> %pred, <vscale x 2 x bfloat> %sub, <vscale x 2 x bfloat> %acc
  ret <vscale x 2 x bfloat> %res
}

define <vscale x 8 x bfloat> @fadd_sel_nxv8bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x i1> %mask) {
; CHECK-LABEL: fadd_sel_nxv8bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    bfadd z1.h, z0.h, z1.h
; CHECK-NEXT:    mov z0.h, p0/m, z1.h
; CHECK-NEXT:    ret
  %sel = select <vscale x 8 x i1> %mask, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> zeroinitializer
  %fadd = fadd nsz <vscale x 8 x bfloat> %a, %sel
  ret <vscale x 8 x bfloat> %fadd
}

define <vscale x 8 x bfloat> @fsub_sel_nxv8bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x i1> %mask) {
; CHECK-LABEL: fsub_sel_nxv8bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    bfsub z1.h, z0.h, z1.h
; CHECK-NEXT:    mov z0.h, p0/m, z1.h
; CHECK-NEXT:    ret
  %sel = select <vscale x 8 x i1> %mask, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> zeroinitializer
  %fsub = fsub <vscale x 8 x bfloat> %a, %sel
  ret <vscale x 8 x bfloat> %fsub
}

define <vscale x 8 x bfloat> @fadd_sel_negzero_nxv8bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x i1> %mask) {
; CHECK-LABEL: fadd_sel_negzero_nxv8bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    bfadd z1.h, z0.h, z1.h
; CHECK-NEXT:    mov z0.h, p0/m, z1.h
; CHECK-NEXT:    ret
  %nz = fneg <vscale x 8 x bfloat> zeroinitializer
  %sel = select <vscale x 8 x i1> %mask, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %nz
  %fadd = fadd <vscale x 8 x bfloat> %a, %sel
  ret <vscale x 8 x bfloat> %fadd
}

define <vscale x 8 x bfloat> @fsub_sel_negzero_nxv8bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x i1> %mask) {
; CHECK-LABEL: fsub_sel_negzero_nxv8bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    bfsub z1.h, z0.h, z1.h
; CHECK-NEXT:    mov z0.h, p0/m, z1.h
; CHECK-NEXT:    ret
  %nz = fneg <vscale x 8 x bfloat> zeroinitializer
  %sel = select <vscale x 8 x i1> %mask, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %nz
  %fsub = fsub nsz <vscale x 8 x bfloat> %a, %sel
  ret <vscale x 8 x bfloat> %fsub
}

define <vscale x 8 x bfloat> @fadd_sel_fmul_nxv8bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, <vscale x 8 x i1> %mask) {
; CHECK-LABEL: fadd_sel_fmul_nxv8bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    movi v3.2d, #0000000000000000
; CHECK-NEXT:    bfmul z1.h, z1.h, z2.h
; CHECK-NEXT:    sel z1.h, p0, z1.h, z3.h
; CHECK-NEXT:    bfadd z0.h, z0.h, z1.h
; CHECK-NEXT:    ret
  %fmul = fmul <vscale x 8 x bfloat> %b, %c
  %sel = select <vscale x 8 x i1> %mask, <vscale x 8 x bfloat> %fmul, <vscale x 8 x bfloat> zeroinitializer
  %fadd = fadd contract <vscale x 8 x bfloat> %a, %sel
  ret <vscale x 8 x bfloat> %fadd
}

define <vscale x 8 x bfloat> @fsub_sel_fmul_nxv8bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, <vscale x 8 x i1> %mask) {
; CHECK-LABEL: fsub_sel_fmul_nxv8bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    bfmul z1.h, z1.h, z2.h
; CHECK-NEXT:    bfsub z1.h, z0.h, z1.h
; CHECK-NEXT:    mov z0.h, p0/m, z1.h
; CHECK-NEXT:    ret
  %fmul = fmul <vscale x 8 x bfloat> %b, %c
  %sel = select <vscale x 8 x i1> %mask, <vscale x 8 x bfloat> %fmul, <vscale x 8 x bfloat> zeroinitializer
  %fsub = fsub contract <vscale x 8 x bfloat> %a, %sel
  ret <vscale x 8 x bfloat> %fsub
}

define <vscale x 8 x bfloat> @fadd_sel_fmul_nsz_nxv8bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, <vscale x 8 x i1> %mask) {
; CHECK-LABEL: fadd_sel_fmul_nsz_nxv8bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    bfmul z1.h, z1.h, z2.h
; CHECK-NEXT:    bfadd z1.h, z0.h, z1.h
; CHECK-NEXT:    mov z0.h, p0/m, z1.h
; CHECK-NEXT:    ret
  %fmul = fmul <vscale x 8 x bfloat> %b, %c
  %sel = select <vscale x 8 x i1> %mask, <vscale x 8 x bfloat> %fmul, <vscale x 8 x bfloat> zeroinitializer
  %fadd = fadd nsz contract <vscale x 8 x bfloat> %a, %sel
  ret <vscale x 8 x bfloat> %fadd
}

define <vscale x 8 x bfloat> @fsub_sel_fmul_nsz_nxv8bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, <vscale x 8 x i1> %mask) {
; CHECK-LABEL: fsub_sel_fmul_nsz_nxv8bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    bfmul z1.h, z1.h, z2.h
; CHECK-NEXT:    bfsub z1.h, z0.h, z1.h
; CHECK-NEXT:    mov z0.h, p0/m, z1.h
; CHECK-NEXT:    ret
  %fmul = fmul <vscale x 8 x bfloat> %b, %c
  %sel = select <vscale x 8 x i1> %mask, <vscale x 8 x bfloat> %fmul, <vscale x 8 x bfloat> zeroinitializer
  %fsub = fsub nsz contract <vscale x 8 x bfloat> %a, %sel
  ret <vscale x 8 x bfloat> %fsub
}

define <vscale x 8 x bfloat> @fadd_sel_fmul_negzero_nxv8bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, <vscale x 8 x i1> %mask) {
; CHECK-LABEL: fadd_sel_fmul_negzero_nxv8bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    bfmul z1.h, z1.h, z2.h
; CHECK-NEXT:    bfadd z1.h, z0.h, z1.h
; CHECK-NEXT:    mov z0.h, p0/m, z1.h
; CHECK-NEXT:    ret
  %fmul = fmul <vscale x 8 x bfloat> %b, %c
  %nz = fneg <vscale x 8 x bfloat> zeroinitializer
  %sel = select <vscale x 8 x i1> %mask, <vscale x 8 x bfloat> %fmul, <vscale x 8 x bfloat> %nz
  %fadd = fadd contract <vscale x 8 x bfloat> %a, %sel
  ret <vscale x 8 x bfloat> %fadd
}

define <vscale x 8 x bfloat> @fsub_sel_fmul_negzero_nxv8bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, <vscale x 8 x i1> %mask) {
; CHECK-LABEL: fsub_sel_fmul_negzero_nxv8bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    mov w8, #32768 // =0x8000
; CHECK-NEXT:    bfmul z1.h, z1.h, z2.h
; CHECK-NEXT:    fmov h3, w8
; CHECK-NEXT:    mov z3.h, h3
; CHECK-NEXT:    sel z1.h, p0, z1.h, z3.h
; CHECK-NEXT:    bfsub z0.h, z0.h, z1.h
; CHECK-NEXT:    ret
  %fmul = fmul <vscale x 8 x bfloat> %b, %c
  %nz = fneg <vscale x 8 x bfloat> zeroinitializer
  %sel = select <vscale x 8 x i1> %mask, <vscale x 8 x bfloat> %fmul, <vscale x 8 x bfloat> %nz
  %fsub = fsub contract <vscale x 8 x bfloat> %a, %sel
  ret <vscale x 8 x bfloat> %fsub
}

define <vscale x 8 x bfloat> @fadd_sel_fmul_negzero_nsz_nxv8bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, <vscale x 8 x i1> %mask) {
; CHECK-LABEL: fadd_sel_fmul_negzero_nsz_nxv8bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    bfmul z1.h, z1.h, z2.h
; CHECK-NEXT:    bfadd z1.h, z0.h, z1.h
; CHECK-NEXT:    mov z0.h, p0/m, z1.h
; CHECK-NEXT:    ret
  %fmul = fmul <vscale x 8 x bfloat> %b, %c
  %nz = fneg <vscale x 8 x bfloat> zeroinitializer
  %sel = select <vscale x 8 x i1> %mask, <vscale x 8 x bfloat> %fmul, <vscale x 8 x bfloat> %nz
  %fadd = fadd nsz contract <vscale x 8 x bfloat> %a, %sel
  ret <vscale x 8 x bfloat> %fadd
}

define <vscale x 8 x bfloat> @fsub_sel_fmul_negzero_nsz_nxv8bf16(<vscale x 8 x bfloat> %a, <vscale x 8 x bfloat> %b, <vscale x 8 x bfloat> %c, <vscale x 8 x i1> %mask) {
; CHECK-LABEL: fsub_sel_fmul_negzero_nsz_nxv8bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    bfmul z1.h, z1.h, z2.h
; CHECK-NEXT:    bfsub z1.h, z0.h, z1.h
; CHECK-NEXT:    mov z0.h, p0/m, z1.h
; CHECK-NEXT:    ret
  %fmul = fmul <vscale x 8 x bfloat> %b, %c
  %nz = fneg <vscale x 8 x bfloat> zeroinitializer
  %sel = select <vscale x 8 x i1> %mask, <vscale x 8 x bfloat> %fmul, <vscale x 8 x bfloat> %nz
  %fsub = fsub nsz contract <vscale x 8 x bfloat> %a, %sel
  ret <vscale x 8 x bfloat> %fsub
}

declare <vscale x 8 x bfloat> @llvm.fma.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>)
