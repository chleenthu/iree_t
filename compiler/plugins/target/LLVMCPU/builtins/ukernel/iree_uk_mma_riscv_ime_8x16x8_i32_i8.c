// compiler/plugins/target/LLVMCPU/builtins/ukernel/iree_uk_mma_riscv_ime_8x16x8_i32_i8.c
#include <riscv_vector.h>
#include "common.h"

// Microkernel for `iree_codegen.inner_tiled` with
// `#iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_IME_8x16x8_I32_I8>`.
// Function name matches the intrinsic name verbatim (lowercased, with the
// `iree_uk_` prefix), same convention as the x86 / AMDGPU / RISC-V "V" C
// ukernels.
//
// Target: SpaceMiT IME "Vector Dot Product Extension" (`+xsmtvdot`). There are
// no C intrinsics for `smt.vmadot` yet, so it is emitted via inline asm, the
// same way the runtime mmt4d ukernel (mmt4d_riscv_64_xsmtvdot.c) does.
//
//   smt.vmadot vd, vs1, vs2   (SEW=8, vl=32 selects the atom=4 MAC unit)
//     C[m,n] += sum_k A[m,k] * B[n,k]   for m,n in [0,4), k in [0,8)   (s8*s8)
//     vs1  = A, int8, (4, 8) row-major        -> byte (m*8 + k)
//     vs2  = B, int8, (4, 8) row-major        -> byte (n*8 + k)
//     vd:vd+1 = C, int32, (4, 4) row-major and contiguous in the register pair
//               -> lane (m*4 + n)   (LMUL=2 pair; vd even.)
//
// The ISA op is a fixed 4x4x8 matrix MAC. This ukernel's *intrinsic tile* is
// 8x16x8: one call over the whole tile issues a 2x4 grid of `smt.vmadot`
// atoms (M: 8/4 = 2, N: 16/4 = 4, K: 8/8 = 1), i.e. 8 back-to-back
// `smt.vmadot` per (outer-K, intrinsics-K) step. That 8-row x 16-col block
// pins 16 ACC vregs (8 LMUL=2 pairs) + 2 LHS + 4 RHS = 22 vregs, so the cost
// model hands this ukernel `intrinsics_m == intrinsics_n == 1` and the
// `intrinsics_{m,n}` loops degenerate to a single iteration — structurally
// identical to iree_uk_mma_riscv_v_7x32x1_f32_f32.c.
//
// Data-tiling pads the matmul M/N/K up to multiples of
// (M0*intrinsics_m) / (N0*intrinsics_n) / (K0*intrinsics_k), so every tile
// handed to this ukernel is full 8x16x8 -- there is no partial-atom or
// partial-grid case, hence no scalar epilogue (same as the "V" ukernel).
//
// ABI (identical to the sibling RISC-V / x86 inner_tiled ukernels): each
// shaped operand is (base pointer, element offset); offsets are in operand
// elements (i8 for LHS/RHS, i32 for ACC). No strides: every tile is
// contiguous. The scalar tail args are the outer-K trip count and the three
// unroll factors; they are compile-time constants at the (always-inlined)
// call site, so every loop below fully unrolls.
//
// Per-intrinsic fragment layouts produced by the data-tiling swizzle for a
// row-major {M0, N0, K0} = {8, 16, 8} intrinsic:
//   - LHS fragment (im, ik): 8x8 i8 row-major panel  (64 bytes)  -> byte m*8+k
//   - RHS fragment (in, ik): 16x8 i8 row-major panel (128 bytes) -> byte n*8+k
//       (the inner_tiled RHS is pre-transposed (N, K), so a row-major (N0, K0)
//        panel already has B laid out N-outer / K-inner as smt.vmadot wants)
//   - ACC fragment (im, in): 8x16 i32 row-major (128 i32) -> lane m*16+n
//        at acc + (im*intrinsics_n + in) * 128
// Packed operand nesting (cross-intrinsic factors outside the panel, K
// innermost, matching the "V" ukernel):
//   LHS: lhs + ((ko*intrinsics_m + im)*intrinsics_k + ik) * 64   bytes
//   RHS: rhs + ((ko*intrinsics_n + in)*intrinsics_k + ik) * 128  bytes
//   ACC: acc + (im*intrinsics_n + in) * 128                      i32
//
// Within one (im, in): the 8x16 row-major ACC fragment is a 2x4 grid of 4x4
// row-major sub-blocks. `smt.vmadot`'s vd:vd+1 output is a 4x4 row-major
// block contiguous in the register pair, but sub-block (mi, ni)'s four rows
// are strided by N0 = 16 in the packed ACC. The load/store uses an indexed
// `vluxei32`/`vsuxei32` with a byte-offset index (`ime_acc_index`) that folds
// that row stride in — the same approach as mmt4d_riscv_64_xsmtvdot.c. This
// also matters for correctness on the X60: a plain unit-stride `vle32`/`vse32`
// at vl == VLMAX gets lowered to `vl2r.v`/`vs2r.v`, which trap as illegal on
// this core; the indexed forms never lower to whole-register moves.
//
// The 8 accumulator pairs stay register-resident across the whole K
// reduction.

enum {
  kImeM0 = 8,
  kImeN0 = 16,
  kImeK0 = 8,
  kImeAtom = 4,                             // ISA op is 4x4x8
  kImeMB = kImeM0 / kImeAtom,               // 2 M sub-atoms
  kImeNB = kImeN0 / kImeAtom,               // 4 N sub-atoms
  kImeLog2Atom = 2,
  kImeLhsPanelBytes = kImeM0 * kImeK0,      // 64
  kImeRhsPanelBytes = kImeN0 * kImeK0,      // 128
  kImeAccFragElems = kImeM0 * kImeN0,       // 128
  kImeSubPanelBytes = kImeAtom * kImeK0,    // 32  (one vle8 at vl=32)
  kImeSubAccElems = kImeAtom * kImeAtom,    // 16  (one vint32m2_t pair)
};

// Byte-offset index mapping accumulator lane i (row-major 4x4, i = r*4 + c) to
// its slot in the strided 8x16 ACC fragment: elem offset r*N0 + c, i.e.
// i + (N0 - 4) * (i / 4); <<2 for bytes.
static IREE_UK_ALWAYS_INLINE vuint32m2_t iree_uk_mma_riscv_ime_acc_index(void) {
  vuint32m2_t vi = __riscv_vid_v_u32m2(kImeSubAccElems);
  vuint32m2_t row = __riscv_vsrl_vx_u32m2(vi, kImeLog2Atom, kImeSubAccElems);
  vuint32m2_t off = __riscv_vmacc_vx_u32m2(vi, (uint32_t)(kImeN0 - kImeAtom),
                                           row, kImeSubAccElems);
  return __riscv_vsll_vx_u32m2(off, 2, kImeSubAccElems);
}

IREE_UK_ALWAYS_INLINE
void iree_uk_mma_riscv_ime_8x16x8_i32_i8(
    const void *lhs_base, int64_t lhs_offset, const void *rhs_base,
    int64_t rhs_offset, void *acc_base, int64_t acc_offset, int32_t k_outer,
    int32_t intrinsics_m, int32_t intrinsics_n, int32_t intrinsics_k) {
  const int8_t *lhs = (const int8_t *)lhs_base + lhs_offset;
  const int8_t *rhs = (const int8_t *)rhs_base + rhs_offset;
  int32_t *acc = (int32_t *)acc_base + acc_offset;

  // vl=32 at SEW=8 is what selects the 4x4x8 (atom=4) IME MAC unit.
  const size_t vl_i8 = kImeSubPanelBytes;
  const size_t vl_acc = kImeSubAccElems;
  const vuint32m2_t acc_idx = iree_uk_mma_riscv_ime_acc_index();

  // Byte stride between consecutive intrinsics-M / intrinsics-N panels, and
  // between consecutive outer-K steps, within the packed operands.
  const int64_t lhs_im_stride = (int64_t)intrinsics_k * kImeLhsPanelBytes;
  const int64_t rhs_in_stride = (int64_t)intrinsics_k * kImeRhsPanelBytes;
  const int64_t lhs_ko_stride = (int64_t)intrinsics_m * lhs_im_stride;
  const int64_t rhs_ko_stride = (int64_t)intrinsics_n * rhs_in_stride;

  for (int32_t im = 0; im < intrinsics_m; ++im) {
    for (int32_t in = 0; in < intrinsics_n; ++in) {
      int32_t *frag =
          acc + (int64_t)(im * intrinsics_n + in) * kImeAccFragElems;
      // Base of sub-block (mi, ni) within the row-major 8x16 fragment.
#define IME_SUB(mi, ni) (frag + ((mi) * kImeAtom) * kImeN0 + (ni) * kImeAtom)

      vint32m2_t c00 = __riscv_vluxei32_v_i32m2(IME_SUB(0, 0), acc_idx, vl_acc);
      vint32m2_t c01 = __riscv_vluxei32_v_i32m2(IME_SUB(0, 1), acc_idx, vl_acc);
      vint32m2_t c02 = __riscv_vluxei32_v_i32m2(IME_SUB(0, 2), acc_idx, vl_acc);
      vint32m2_t c03 = __riscv_vluxei32_v_i32m2(IME_SUB(0, 3), acc_idx, vl_acc);
      vint32m2_t c10 = __riscv_vluxei32_v_i32m2(IME_SUB(1, 0), acc_idx, vl_acc);
      vint32m2_t c11 = __riscv_vluxei32_v_i32m2(IME_SUB(1, 1), acc_idx, vl_acc);
      vint32m2_t c12 = __riscv_vluxei32_v_i32m2(IME_SUB(1, 2), acc_idx, vl_acc);
      vint32m2_t c13 = __riscv_vluxei32_v_i32m2(IME_SUB(1, 3), acc_idx, vl_acc);

      const int8_t *lhs_blk = lhs + (int64_t)im * lhs_im_stride;
      const int8_t *rhs_blk = rhs + (int64_t)in * rhs_in_stride;

      for (int32_t ko = 0; ko < k_outer; ++ko) {
        const int8_t *lhs_ko = lhs_blk + (int64_t)ko * lhs_ko_stride;
        const int8_t *rhs_ko = rhs_blk + (int64_t)ko * rhs_ko_stride;
        IREE_UK_UNROLL for (int32_t ik = 0; ik < intrinsics_k; ++ik) {
          const int8_t *lp = lhs_ko + (int64_t)ik * kImeLhsPanelBytes;
          const int8_t *rp = rhs_ko + (int64_t)ik * kImeRhsPanelBytes;
          // LHS row-group mi and RHS col-group ni are contiguous 4x8 = 32-byte
          // sub-panels of the row-major 8x8 / 16x8 packed panels.
          vint8m1_t a0 = __riscv_vle8_v_i8m1(lp + 0 * kImeSubPanelBytes, vl_i8);
          vint8m1_t a1 = __riscv_vle8_v_i8m1(lp + 1 * kImeSubPanelBytes, vl_i8);
          vint8m1_t b0 = __riscv_vle8_v_i8m1(rp + 0 * kImeSubPanelBytes, vl_i8);
          vint8m1_t b1 = __riscv_vle8_v_i8m1(rp + 1 * kImeSubPanelBytes, vl_i8);
          vint8m1_t b2 = __riscv_vle8_v_i8m1(rp + 2 * kImeSubPanelBytes, vl_i8);
          vint8m1_t b3 = __riscv_vle8_v_i8m1(rp + 3 * kImeSubPanelBytes, vl_i8);
          // One vtype set, then 8 back-to-back smt.vmadot over the 2x4 grid.
          __asm__ volatile(
              "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
              "smt.vmadot %[d0], %[a0], %[b0]\n\t"
              "smt.vmadot %[d1], %[a0], %[b1]\n\t"
              "smt.vmadot %[d2], %[a0], %[b2]\n\t"
              "smt.vmadot %[d3], %[a0], %[b3]\n\t"
              "smt.vmadot %[d4], %[a1], %[b0]\n\t"
              "smt.vmadot %[d5], %[a1], %[b1]\n\t"
              "smt.vmadot %[d6], %[a1], %[b2]\n\t"
              "smt.vmadot %[d7], %[a1], %[b3]\n\t"
              : [d0] "+vr"(c00), [d1] "+vr"(c01), [d2] "+vr"(c02),
                [d3] "+vr"(c03), [d4] "+vr"(c10), [d5] "+vr"(c11),
                [d6] "+vr"(c12), [d7] "+vr"(c13)
              : [a0] "vr"(a0), [a1] "vr"(a1), [b0] "vr"(b0), [b1] "vr"(b1),
                [b2] "vr"(b2), [b3] "vr"(b3), [vl] "r"(vl_i8));
        }
      }

      __riscv_vsuxei32_v_i32m2(IME_SUB(0, 0), acc_idx, c00, vl_acc);
      __riscv_vsuxei32_v_i32m2(IME_SUB(0, 1), acc_idx, c01, vl_acc);
      __riscv_vsuxei32_v_i32m2(IME_SUB(0, 2), acc_idx, c02, vl_acc);
      __riscv_vsuxei32_v_i32m2(IME_SUB(0, 3), acc_idx, c03, vl_acc);
      __riscv_vsuxei32_v_i32m2(IME_SUB(1, 0), acc_idx, c10, vl_acc);
      __riscv_vsuxei32_v_i32m2(IME_SUB(1, 1), acc_idx, c11, vl_acc);
      __riscv_vsuxei32_v_i32m2(IME_SUB(1, 2), acc_idx, c12, vl_acc);
      __riscv_vsuxei32_v_i32m2(IME_SUB(1, 3), acc_idx, c13, vl_acc);
#undef IME_SUB
    }
  }
}
