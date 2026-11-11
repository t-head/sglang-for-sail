// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <hggc_bf16.h>
#include <cstdint>

namespace step5_dsa {

constexpr int kHeadDim = 192;
constexpr int kRegionTokens = 8;
constexpr int kCubeWidth = 64;
constexpr int kHeadCubes = kHeadDim / kCubeWidth;
constexpr int kHeadMmaTiles = kHeadDim / 16;
constexpr int64_t kRowBytes = kHeadDim * sizeof(__ppu_bfloat16);
constexpr int64_t kCubeRowBytes = kCubeWidth * sizeof(__ppu_bfloat16);

template <int Regions>
using KvStage = __ppu_bfloat16[Regions][kHeadCubes][kRegionTokens][kCubeWidth];

struct MmaOperand {
  uint32_t x[4];
};

struct MmaAccumulator {
  float x[8];
};

__device__ __forceinline__ void ldm_x4(MmaOperand &f, const __ppu_bfloat16 *p) {
  auto *d = f.x;
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile(
      "ppu.tc02.ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, "
      "[%4];\n\t"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a));
}

__device__ __forceinline__ void ldm_x4_trans(MmaOperand &f, const __ppu_bfloat16 *p) {
  auto *d = f.x;
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile(
      "ppu.tc02.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, "
      "%3}, [%4];\n\t"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a));
}

__device__ __forceinline__ void
aiu_copy_group(__ppu_bfloat16 *dst_shared, const __ppu_bfloat16 *src_tensor,
               int extent, int start, int rows) {
  const unsigned dst =
      static_cast<unsigned>(__cvta_generic_to_shared(dst_shared));
  asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.cs.padz.swzl.b16 "
               "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, "
               "%12}, %13;\n\t"
               :: "r"(dst), "l"(src_tensor), "r"(kCubeWidth), "r"(extent),
                 "r"(kHeadCubes), "r"(kCubeWidth), "r"(rows), "r"(kHeadCubes),
                 "l"(kRowBytes), "l"(kCubeRowBytes), "r"(0), "r"(start), "r"(0),
                 "r"(0) // swzlMode 0 = 128 B
               : "memory");
}

__device__ __forceinline__ float head_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}

__device__ __forceinline__ float head_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

template <class T>
__device__ __forceinline__ void clear_fragment(T &a) {
#pragma unroll
  for (int i = 0; i < sizeof(a.x) / sizeof(a.x[0]); ++i)
    a.x[i] = 0;
}

__device__ __forceinline__ void mma_bf16(MmaAccumulator &d,
    const MmaOperand &a, const MmaOperand &b, const MmaAccumulator &c) {
  asm volatile(
      "ppu.tc02.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, "
      "{%12,%13,%14,%15}, {%16,%17,%18,%19,%20,%21,%22,%23};"
      : "=r"(d.x[0]), "=r"(d.x[1]), "=r"(d.x[2]), "=r"(d.x[3]),
        "=r"(d.x[4]), "=r"(d.x[5]), "=r"(d.x[6]), "=r"(d.x[7])
      : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]),
        "r"(b.x[0]), "r"(b.x[1]), "r"(b.x[2]), "r"(b.x[3]),
        "r"(c.x[0]), "r"(c.x[1]), "r"(c.x[2]), "r"(c.x[3]),
        "r"(c.x[4]), "r"(c.x[5]), "r"(c.x[6]), "r"(c.x[7]));
}

} // namespace step5_dsa
