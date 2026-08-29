// compiler/plugins/target/LLVMCPU/builtins/ukernel/iree_uk_mma_riscv_ime_1x16x8_i32_i8.c
#include <riscv_vector.h>
#include "common.h"

// Narrow-M (M0 = 1) companion to iree_uk_mma_riscv_ime_8x16x8_i32_i8.c, for the
// LLM *decode* step, where the matmul M dim is 1 and the 8x16x8 intrinsic would
// pad it 8x. Same ISA (SpaceMiT IME `smt.vmadot`, feature `+xsmtvdot`), same
// ABI, same packed-operand nesting; only M0 changes 8 -> 1.
//
//   smt.vmadot vd, vs1, vs2   (SEW=8, vl=32 selects the atom=4 MAC unit)
//     C[m,n] += sum_k A[m,k] * B[n,k]   for m,n in [0,4), k in [0,8)   (s8*s8)
//     vs1  = A, int8, (4, 8) row-major        -> byte (m*8 + k)
//     vs2  = B, int8, (4, 8) row-major        -> byte (n*8 + k)
//     vd:vd+1 = C, int32, (4, 4) row-major, contiguous in the register pair
//               -> lane (m*4 + n)   (LMUL=2 pair; vd even.)
//
// The ISA op is still a fixed 4x4x8 MAC. With M0 = 1 only row m = 0 of every
// atom is real; rows 1..3 read don't-care LHS lanes and their results are never
// stored. This still halves the `smt.vmadot` count versus the 8x16x8 tile (one
// M sub-atom instead of two) and shrinks the packed LHS 8x -- the win for a
// memory-bound decode GEMV comes mostly from the latter.
//
// This intrinsic tile is 1x16x8: one call issues a 1x4 grid of `smt.vmadot`
// atoms (M: 1, using 1 of the 4 atom rows; N: 16/4 = 4; K: 8/8 = 1), i.e. 4
// back-to-back `smt.vmadot` per (outer-K, intrinsics-K) step. That pins 4 ACC
// vregs (4 LMUL=2 pairs) + 1 LHS + 4 RHS = well under the file, so the cost
// model degenerates `intrinsics_{m,n} == 1` the same way the 8x16x8 and the
// "V" f32 ukernels do.
//
// Data-tiling pads M/N/K up to multiples of (M0*intrinsics_m) /
// (N0*intrinsics_n) / (K0*intrinsics_k), so every tile handed here is a full
// 1x16x8 -- no partial-atom / partial-grid case, hence no scalar epilogue.
//
// ABI (identical to the sibling inner_tiled ukernels): each shaped operand is
// (base pointer, element offset); offsets are in operand elements (i8 for
// LHS/RHS, i32 for ACC). No strides. The scalar tail args are the outer-K trip
// count and the three unroll factors; all compile-time constants at the
// always-inlined call site, so every loop below fully unrolls.
//
// Per-intrinsic fragment layouts produced by the data-tiling swizzle for a
// row-major {M0, N0, K0} = {1, 16, 8} intrinsic:
//   - LHS fragment (ik): 1x8 i8 row-major panel (8 bytes)   -> byte k
//   - RHS fragment (in, ik): 16x8 i8 row-major panel (128 bytes) -> byte n*8+k
//       (the inner_tiled RHS is pre-transposed (N, K), so a row-major (N0, K0)
//        panel already has B laid out N-outer / K-inner as smt.vmadot wants)
//   - ACC fragment (in): 1x16 i32 row-major (16 i32)         -> lane n
//        at acc + (im*intrinsics_n + in) * 16
// Packed operand nesting (cross-intrinsic factors outside the panel, K
// innermost, matching the 8x16x8 ukernel):
//   LHS: lhs + ((ko*intrinsics_m + im)*intrinsics_k + ik) * 8    bytes
//   RHS: rhs + ((ko*intrinsics_n + in)*intrinsics_k + ik) * 128  bytes
//   ACC: acc + (im*intrinsics_n + in) * 16                       i32
//
// The 1x16 row-major ACC fragment is 4 contiguous 4-wide i32 sub-blocks, one
// per `smt.vmadot` N atom; unlike the 8x16x8 case the sub-block's rows are not
// strided (there is only row 0), so the load/store is a plain 4-lane access.
// Indexed `vluxei32`/`vsuxei32` (byte index [0,4,8,12]) is still used instead
// of unit-stride `vle32`/`vse32` so the store never lowers to `vs2r.v`, which
// traps as illegal on the X60 (see the 8x16x8 ukernel).
//
// The LHS panel is only 8 bytes, so it is loaded with `vle8` at vl = 8 (never
// vl = 32): reading the full 32-byte atom width would over-read up to 24 bytes
// past the packed LHS buffer. `smt.vmadot` still consumes vs1 as 32 bytes from
// the register; lanes 8..31 are the tail (agnostic) and feed only the discarded
// rows 1..3.

enum {
  kImeM0 = 1,
  kImeN0 = 16,
  kImeK0 = 8,
  kImeAtom = 4,                          // ISA op is 4x4x8
  kImeNB = kImeN0 / kImeAtom,            // 4 N sub-atoms
  kImeLog2Atom = 2,
  kImeLhsPanelBytes = kImeM0 * kImeK0,   // 8
  kImeRhsPanelBytes = kImeN0 * kImeK0,   // 128
  kImeAccFragElems = kImeM0 * kImeN0,    // 16
  kImeLhsVl = kImeM0 * kImeK0,           // 8   (vle8 reads exactly the panel)
  kImeSubRhsBytes = kImeAtom * kImeK0,   // 32  (one vle8 at vl=32)
  kImeVmadotVl = kImeAtom * kImeK0,      // 32  (selects the atom=4 MAC unit)
  kImeSubAccElems = kImeAtom,            // 4   (row 0 of one 4x4 atom)
};

// Byte-offset index [0, 4, 8, 12] selecting the 4 contiguous i32 of one atom's
// row 0. (vid * 4; the high lanes of the m2 pair are unused at vl = 4.)
static IREE_UK_ALWAYS_INLINE vuint32m2_t iree_uk_mma_riscv_ime1_acc_index(void) {
  return __riscv_vsll_vx_u32m2(__riscv_vid_v_u32m2(kImeSubAccElems), 2,
                               kImeSubAccElems);
}

IREE_UK_ALWAYS_INLINE
void iree_uk_mma_riscv_ime_1x16x8_i32_i8(
    const void *lhs_base, int64_t lhs_offset, const void *rhs_base,
    int64_t rhs_offset, void *acc_base, int64_t acc_offset, int32_t k_outer,
    int32_t intrinsics_m, int32_t intrinsics_n, int32_t intrinsics_k) {
  const int8_t *lhs = (const int8_t *)lhs_base + lhs_offset;
  const int8_t *rhs = (const int8_t *)rhs_base + rhs_offset;
  int32_t *acc = (int32_t *)acc_base + acc_offset;

  const size_t vl_lhs = kImeLhsVl;
  const size_t vl_rhs = kImeSubRhsBytes;
  const size_t vl_acc = kImeSubAccElems;
  const size_t vl_mac = kImeVmadotVl;
  const vuint32m2_t acc_idx = iree_uk_mma_riscv_ime1_acc_index();

  const int64_t lhs_im_stride = (int64_t)intrinsics_k * kImeLhsPanelBytes;
  const int64_t rhs_in_stride = (int64_t)intrinsics_k * kImeRhsPanelBytes;
  const int64_t lhs_ko_stride = (int64_t)intrinsics_m * lhs_im_stride;
  const int64_t rhs_ko_stride = (int64_t)intrinsics_n * rhs_in_stride;

  const int64_t total_k = (int64_t)k_outer * intrinsics_k;

  for (int32_t im = 0; im < intrinsics_m; ++im) {
    for (int32_t in = 0; in < intrinsics_n; ++in) {
      int32_t *frag =
          acc + (int64_t)(im * intrinsics_n + in) * kImeAccFragElems;
#define IME1_SUB(ni) (frag + (ni) * kImeAtom)

      vint32m2_t c0 = __riscv_vluxei32_v_i32m2(IME1_SUB(0), acc_idx, vl_acc);
      vint32m2_t c1 = __riscv_vluxei32_v_i32m2(IME1_SUB(1), acc_idx, vl_acc);
      vint32m2_t c2 = __riscv_vluxei32_v_i32m2(IME1_SUB(2), acc_idx, vl_acc);
      vint32m2_t c3 = __riscv_vluxei32_v_i32m2(IME1_SUB(3), acc_idx, vl_acc);

      const int8_t *lhs_blk = lhs + (int64_t)im * lhs_im_stride;
      const int8_t *rhs_blk = rhs + (int64_t)in * rhs_in_stride;

      for (int64_t k = 0; k < total_k; ++k) {
        const int32_t ko = (int32_t)(k / intrinsics_k);
        const int32_t ik = (int32_t)(k % intrinsics_k);
        const int8_t *lp = lhs_blk + (int64_t)ko * lhs_ko_stride +
                           (int64_t)ik * kImeLhsPanelBytes;
        const int8_t *rp = rhs_blk + (int64_t)ko * rhs_ko_stride +
                           (int64_t)ik * kImeRhsPanelBytes;
        vint8m1_t a0 = __riscv_vle8_v_i8m1(lp, vl_lhs);
        vint8m1_t b0 = __riscv_vle8_v_i8m1(rp + 0 * kImeSubRhsBytes, vl_rhs);
        vint8m1_t b1 = __riscv_vle8_v_i8m1(rp + 1 * kImeSubRhsBytes, vl_rhs);
        vint8m1_t b2 = __riscv_vle8_v_i8m1(rp + 2 * kImeSubRhsBytes, vl_rhs);
        vint8m1_t b3 = __riscv_vle8_v_i8m1(rp + 3 * kImeSubRhsBytes, vl_rhs);
        __asm__ volatile(
            "vsetvli zero, %[vl], e8, m1, ta, ma\n\t"
            "smt.vmadot %[d0], %[a0], %[b0]\n\t"
            "smt.vmadot %[d1], %[a0], %[b1]\n\t"
            "smt.vmadot %[d2], %[a0], %[b2]\n\t"
            "smt.vmadot %[d3], %[a0], %[b3]\n\t"
            : [d0] "+vr"(c0), [d1] "+vr"(c1), [d2] "+vr"(c2), [d3] "+vr"(c3)
            : [a0] "vr"(a0), [b0] "vr"(b0), [b1] "vr"(b1), [b2] "vr"(b2),
              [b3] "vr"(b3), [vl] "r"(vl_mac));
      }

      __riscv_vsuxei32_v_i32m2(IME1_SUB(0), acc_idx, c0, vl_acc);
      __riscv_vsuxei32_v_i32m2(IME1_SUB(1), acc_idx, c1, vl_acc);
      __riscv_vsuxei32_v_i32m2(IME1_SUB(2), acc_idx, c2, vl_acc);
      __riscv_vsuxei32_v_i32m2(IME1_SUB(3), acc_idx, c3, vl_acc);
#undef IME1_SUB
    }
  }
}
