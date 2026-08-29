// compiler/plugins/target/LLVMCPU/builtins/ukernel/iree_uk_mma_riscv_v_1x32x1_f32_f32.c
#include <riscv_vector.h>
#include "common.h"
// ABI matches the x86 ukernels: base+offset pointers (contiguous data-tiled
// layout, no strides), intrinsics_{m,n,k} as always-inline-specialized args.
//   - ACC: one vfloat32m4_t (M0=1 x N0=32 f32) per (m,n), packed row-major:
//     fragment (m,n) at acc + (m*intrinsics_n + n) * 32.
//   - LHS: per outer-K step, intrinsics_m*intrinsics_k f32 scalars, [m][k].
//   - RHS: per outer-K step, intrinsics_n*intrinsics_k panels of 32 f32, [n][k].
IREE_UK_ALWAYS_INLINE
void iree_uk_mma_riscv_v_1x32x1_f32_f32(
    const float *lhs_base, int64_t lhs_offset, const float *rhs_base,
    int64_t rhs_offset, float *acc_base, int64_t acc_offset, int32_t k_outer,
    int32_t intrinsics_m, int32_t intrinsics_n, int32_t intrinsics_k) {
  const size_t vl = 32;  // Fixed atomic N=32.
  // ASSUMPTION: verify against sibling ukernel -- treating k_outer as the
  // reduction trip count and intrinsics_k as an inner multiplier (normally 1
  // for a native K=1 tile depth).
  const int64_t total_k = (int64_t)k_outer * intrinsics_k;
  // ASSUMPTION: acc tile is row-major with (1*intrinsics_m) rows and
  // (32*intrinsics_n) columns; row stride is therefore 32*intrinsics_n.
  const int64_t acc_row_stride = (int64_t)intrinsics_n * vl;
  for (int32_t im = 0; im < intrinsics_m; ++im) {
    for (int32_t in = 0; in < intrinsics_n; ++in) {
      const float *lhs_ptr =
          lhs_base + lhs_offset + (int64_t)im * 1 * total_k;
      const float *rhs_ptr =
          rhs_base + rhs_offset + (int64_t)in * vl * total_k;
      float *out_ptr = acc_base + acc_offset + (int64_t)im * 1 * acc_row_stride
                        + (int64_t)in * vl;
      vfloat32m4_t acc0;
      acc0 = __riscv_vle32_v_f32m4(out_ptr, vl);
      for (int64_t k = 0; k < total_k; ++k) {
        vfloat32m4_t rhs = __riscv_vle32_v_f32m4(rhs_ptr, vl);
        rhs_ptr += vl;
        float lhs0 = *lhs_ptr++;
        acc0 = __riscv_vfmacc_vf_f32m4(acc0, lhs0, rhs, vl);
      }
      __riscv_vse32_v_f32m4(out_ptr, acc0, vl);
    }
  }
}
