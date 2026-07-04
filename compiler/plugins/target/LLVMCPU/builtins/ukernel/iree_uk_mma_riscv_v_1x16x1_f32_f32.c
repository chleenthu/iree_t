// compiler/plugins/target/LLVMCPU/builtins/ukernel/iree_uk_mma_riscv_v_1x16x1_f32_f32.c
#include <riscv_vector.h>
#include "common.h"

// ABI matches the x86 ukernels: base+offset pointers (contiguous data-tiled
// layout, no strides), intrinsics_{m,n,k} as always-inline-specialized args.
//   - ACC: one vfloat32m4_t (M0=1 x N0=16 f32) per (m,n), packed row-major:
//     fragment (m,n) at acc + (m*intrinsics_n + n) * 16.
//   - LHS: per outer-K step, intrinsics_m*intrinsics_k f32 scalars, [m][k].
//   - RHS: per outer-K step, intrinsics_n*intrinsics_k panels of 16 f32, [n][k].

#define DECLARE_ACC_1 vfloat32m4_t acc0;
#define DECLARE_ACC_2 DECLARE_ACC_1 vfloat32m4_t acc1;
#define DECLARE_ACC_3 DECLARE_ACC_2 vfloat32m4_t acc2;
#define DECLARE_ACC_4 DECLARE_ACC_3 vfloat32m4_t acc3;
#define DECLARE_ACC_5 DECLARE_ACC_4 vfloat32m4_t acc4;
#define DECLARE_ACC_6 DECLARE_ACC_5 vfloat32m4_t acc5;
#define DECLARE_ACC_7 DECLARE_ACC_6 vfloat32m4_t acc6;

#define LOAD_ACC_1 acc0 = __riscv_vle32_v_f32m4(acc + (0 * intrinsics_n + n) * kAccFragElems, vl);
#define LOAD_ACC_2 LOAD_ACC_1 acc1 = __riscv_vle32_v_f32m4(acc + (1 * intrinsics_n + n) * kAccFragElems, vl);
#define LOAD_ACC_3 LOAD_ACC_2 acc2 = __riscv_vle32_v_f32m4(acc + (2 * intrinsics_n + n) * kAccFragElems, vl);
#define LOAD_ACC_4 LOAD_ACC_3 acc3 = __riscv_vle32_v_f32m4(acc + (3 * intrinsics_n + n) * kAccFragElems, vl);
#define LOAD_ACC_5 LOAD_ACC_4 acc4 = __riscv_vle32_v_f32m4(acc + (4 * intrinsics_n + n) * kAccFragElems, vl);
#define LOAD_ACC_6 LOAD_ACC_5 acc5 = __riscv_vle32_v_f32m4(acc + (5 * intrinsics_n + n) * kAccFragElems, vl);
#define LOAD_ACC_7 LOAD_ACC_6 acc6 = __riscv_vle32_v_f32m4(acc + (6 * intrinsics_n + n) * kAccFragElems, vl);

#define STORE_ACC_1 __riscv_vse32_v_f32m4(acc + (0 * intrinsics_n + n) * kAccFragElems, acc0, vl);
#define STORE_ACC_2 STORE_ACC_1 __riscv_vse32_v_f32m4(acc + (1 * intrinsics_n + n) * kAccFragElems, acc1, vl);
#define STORE_ACC_3 STORE_ACC_2 __riscv_vse32_v_f32m4(acc + (2 * intrinsics_n + n) * kAccFragElems, acc2, vl);
#define STORE_ACC_4 STORE_ACC_3 __riscv_vse32_v_f32m4(acc + (3 * intrinsics_n + n) * kAccFragElems, acc3, vl);
#define STORE_ACC_5 STORE_ACC_4 __riscv_vse32_v_f32m4(acc + (4 * intrinsics_n + n) * kAccFragElems, acc4, vl);
#define STORE_ACC_6 STORE_ACC_5 __riscv_vse32_v_f32m4(acc + (5 * intrinsics_n + n) * kAccFragElems, acc5, vl);
#define STORE_ACC_7 STORE_ACC_6 __riscv_vse32_v_f32m4(acc + (6 * intrinsics_n + n) * kAccFragElems, acc6, vl);

#define FMA_ACC_1 acc0 = __riscv_vfmacc_vf_f32m4(acc0, lhs[lhs_idx + 0 * intrinsics_k + k], rhs_panel, vl);
#define FMA_ACC_2 FMA_ACC_1 acc1 = __riscv_vfmacc_vf_f32m4(acc1, lhs[lhs_idx + 1 * intrinsics_k + k], rhs_panel, vl);
#define FMA_ACC_3 FMA_ACC_2 acc2 = __riscv_vfmacc_vf_f32m4(acc2, lhs[lhs_idx + 2 * intrinsics_k + k], rhs_panel, vl);
#define FMA_ACC_4 FMA_ACC_3 acc3 = __riscv_vfmacc_vf_f32m4(acc3, lhs[lhs_idx + 3 * intrinsics_k + k], rhs_panel, vl);
#define FMA_ACC_5 FMA_ACC_4 acc4 = __riscv_vfmacc_vf_f32m4(acc4, lhs[lhs_idx + 4 * intrinsics_k + k], rhs_panel, vl);
#define FMA_ACC_6 FMA_ACC_5 acc5 = __riscv_vfmacc_vf_f32m4(acc5, lhs[lhs_idx + 5 * intrinsics_k + k], rhs_panel, vl);
#define FMA_ACC_7 FMA_ACC_6 acc6 = __riscv_vfmacc_vf_f32m4(acc6, lhs[lhs_idx + 6 * intrinsics_k + k], rhs_panel, vl);

#define IREE_UK_MMA_RISCV_V_UNROLLED_M(M_VAL) \
  case M_VAL: { \
    for (int32_t n = 0; n < intrinsics_n; ++n) { \
      DECLARE_ACC_##M_VAL \
      LOAD_ACC_##M_VAL \
      for (int32_t ko = 0; ko < k_outer; ++ko) { \
        for (int32_t k = 0; k < intrinsics_k; ++k) { \
          vfloat32m4_t rhs_panel = __riscv_vle32_v_f32m4( \
              rhs + ((int64_t)ko * intrinsics_n * intrinsics_k + \
                     n * intrinsics_k + k) * kAccFragElems, vl); \
          int64_t lhs_idx = (int64_t)ko * intrinsics_m * intrinsics_k; \
          FMA_ACC_##M_VAL \
        } \
      } \
      STORE_ACC_##M_VAL \
    } \
    break; \
  }


IREE_UK_ALWAYS_INLINE
void iree_uk_mma_riscv_v_1x16x1_f32_f32(
    const float *lhs_base, int64_t lhs_offset, const float *rhs_base,
    int64_t rhs_offset, float *acc_base, int64_t acc_offset, int32_t k_outer,
    int32_t intrinsics_m, int32_t intrinsics_n, int32_t intrinsics_k) {

  const float *lhs = lhs_base + lhs_offset;
  const float *rhs = rhs_base + rhs_offset;
  float *acc = acc_base + acc_offset;
  enum { kAccFragElems = 16 };
  size_t vl = kAccFragElems;

  switch (intrinsics_m) {
    IREE_UK_MMA_RISCV_V_UNROLLED_M(1)
    IREE_UK_MMA_RISCV_V_UNROLLED_M(2)
    IREE_UK_MMA_RISCV_V_UNROLLED_M(3)
    IREE_UK_MMA_RISCV_V_UNROLLED_M(4)
    IREE_UK_MMA_RISCV_V_UNROLLED_M(5)
    IREE_UK_MMA_RISCV_V_UNROLLED_M(6)
    IREE_UK_MMA_RISCV_V_UNROLLED_M(7)
    default: {
      for (int32_t m = 0; m < intrinsics_m; ++m) {
        for (int32_t n = 0; n < intrinsics_n; ++n) {
          vfloat32m4_t acc_reg = __riscv_vle32_v_f32m4(
              acc + (m * intrinsics_n + n) * kAccFragElems, vl);
          for (int32_t ko = 0; ko < k_outer; ++ko) {
            for (int32_t k = 0; k < intrinsics_k; ++k) {
              float lhs_scalar =
                  lhs[(int64_t)ko * intrinsics_m * intrinsics_k +
                      m * intrinsics_k + k];
              vfloat32m4_t rhs_panel = __riscv_vle32_v_f32m4(
                  rhs + ((int64_t)ko * intrinsics_n * intrinsics_k +
                         n * intrinsics_k + k) *
                            kAccFragElems,
                  vl);
              acc_reg = __riscv_vfmacc_vf_f32m4(acc_reg, lhs_scalar, rhs_panel, vl);
            }
          }
          __riscv_vse32_v_f32m4(acc + (m * intrinsics_n + n) * kAccFragElems,
                                acc_reg, vl);
        }
      }
      break;
    }
  }
}
/*IREE_UK_ALWAYS_INLINE
void iree_uk_mma_riscv_v_1x16x1_f32_f32(
    const float *lhs_base, int64_t lhs_offset, const float *rhs_base,
    int64_t rhs_offset, float *acc_base, int64_t acc_offset, int32_t k_outer,
    int32_t intrinsics_m, int32_t intrinsics_n, int32_t intrinsics_k) {
  const float *lhs = lhs_base + lhs_offset;
  const float *rhs = rhs_base + rhs_offset;
  float *acc = acc_base + acc_offset;
  enum { kAccFragElems = 16 };
  size_t vl = kAccFragElems;

  for (int32_t m = 0; m < intrinsics_m; ++m) {
    for (int32_t n = 0; n < intrinsics_n; ++n) {
      vfloat32m4_t acc_reg = __riscv_vle32_v_f32m4(
          acc + (m * intrinsics_n + n) * kAccFragElems, vl);
      for (int32_t ko = 0; ko < k_outer; ++ko) {
        for (int32_t k = 0; k < intrinsics_k; ++k) {
          float lhs_scalar =
              lhs[(int64_t)ko * intrinsics_m * intrinsics_k +
                  m * intrinsics_k + k];
          vfloat32m4_t rhs_panel = __riscv_vle32_v_f32m4(
              rhs + ((int64_t)ko * intrinsics_n * intrinsics_k +
                     n * intrinsics_k + k) *
                        kAccFragElems,
              vl);
          acc_reg = __riscv_vfmacc_vf_f32m4(acc_reg, lhs_scalar, rhs_panel, vl);
        }
      }
      __riscv_vse32_v_f32m4(acc + (m * intrinsics_n + n) * kAccFragElems,
                            acc_reg, vl);
    }
  }
}*/
