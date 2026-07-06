// compiler/plugins/target/LLVMCPU/builtins/ukernel/iree_uk_mma_riscv_v_7x32x1_f32_f32.c
#include <riscv_vector.h>
#include "common.h"

// ABI matches the x86 ukernels: base+offset pointers (contiguous data-tiled
// layout, no strides), intrinsics_{m,n,k} as always-inline-specialized args.
//   - ACC: one vfloat32m4_t (M0=1 x N0=16 f32) per (m,n), packed row-major:
//     fragment (m,n) at acc + (m*intrinsics_n + n) * 16.
//   - LHS: per outer-K step, intrinsics_m*intrinsics_k f32 scalars, [m][k].
//   - RHS: per outer-K step, intrinsics_n*intrinsics_k panels of 16 f32, [n][k].

IREE_UK_ALWAYS_INLINE
void iree_uk_mma_riscv_v_7x32x1_f32_f32(
    const float *lhs_base, int64_t lhs_offset, const float *rhs_base,
    int64_t rhs_offset, float *acc_base, int64_t acc_offset, int32_t k_outer,
    int32_t intrinsics_m, int32_t intrinsics_n, int32_t intrinsics_k) {

  const size_t vl = 32;  // Fixed atomic N=32.
  // ASSUMPTION: verify against sibling ukernel -- treating k_outer as the
  // reduction trip count and intrinsics_k as an inner multiplier (normally 1
  // for a native K=1 tile depth).
  const int64_t total_k = (int64_t)k_outer * intrinsics_k;
  // ASSUMPTION: acc tile is row-major with (7*intrinsics_m) rows and
  // (32*intrinsics_n) columns; row stride is therefore 32*intrinsics_n.
  const int64_t acc_row_stride = (int64_t)intrinsics_n * vl;

  for (int32_t im = 0; im < intrinsics_m; ++im) {
    for (int32_t in = 0; in < intrinsics_n; ++in) {
      const float *lhs_ptr =
          lhs_base + lhs_offset + (int64_t)im * 7 * total_k;
      const float *rhs_ptr =
          rhs_base + rhs_offset + (int64_t)in * vl * total_k;
      float *out_ptr = acc_base + acc_offset + (int64_t)im * 7 * acc_row_stride
                        + (int64_t)in * vl;

      vfloat32m4_t acc0, acc1, acc2, acc3, acc4, acc5, acc6;
      acc0 = __riscv_vle32_v_f32m4(out_ptr, vl);
      acc1 = __riscv_vle32_v_f32m4(out_ptr + acc_row_stride, vl);
      acc2 = __riscv_vle32_v_f32m4(out_ptr + acc_row_stride * 2, vl);
      acc3 = __riscv_vle32_v_f32m4(out_ptr + acc_row_stride * 3, vl);
      acc4 = __riscv_vle32_v_f32m4(out_ptr + acc_row_stride * 4, vl);
      acc5 = __riscv_vle32_v_f32m4(out_ptr + acc_row_stride * 5, vl);
      acc6 = __riscv_vle32_v_f32m4(out_ptr + acc_row_stride * 6, vl);

      for (int64_t k = 0; k < total_k; ++k) {
        vfloat32m4_t rhs = __riscv_vle32_v_f32m4(rhs_ptr, vl);
        rhs_ptr += vl;
        float lhs[7];
        IREE_UK_UNROLL for (int i = 0; i < 7; ++i) { lhs[i] = *lhs_ptr++; }
        acc0 = __riscv_vfmacc_vf_f32m4(acc0, lhs[0], rhs, vl);
        acc1 = __riscv_vfmacc_vf_f32m4(acc1, lhs[1], rhs, vl);
        acc2 = __riscv_vfmacc_vf_f32m4(acc2, lhs[2], rhs, vl);
        acc3 = __riscv_vfmacc_vf_f32m4(acc3, lhs[3], rhs, vl);
        acc4 = __riscv_vfmacc_vf_f32m4(acc4, lhs[4], rhs, vl);
        acc5 = __riscv_vfmacc_vf_f32m4(acc5, lhs[5], rhs, vl);
        acc6 = __riscv_vfmacc_vf_f32m4(acc6, lhs[6], rhs, vl);
      }

      __riscv_vse32_v_f32m4(out_ptr, acc0, vl);
      __riscv_vse32_v_f32m4(out_ptr + acc_row_stride, acc1, vl);
      __riscv_vse32_v_f32m4(out_ptr + acc_row_stride * 2, acc2, vl);
      __riscv_vse32_v_f32m4(out_ptr + acc_row_stride * 3, acc3, vl);
      __riscv_vse32_v_f32m4(out_ptr + acc_row_stride * 4, acc4, vl);
      __riscv_vse32_v_f32m4(out_ptr + acc_row_stride * 5, acc5, vl);
      __riscv_vse32_v_f32m4(out_ptr + acc_row_stride * 6, acc6, vl);
    }
  }
}
