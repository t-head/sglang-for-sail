// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include <hggc_bf16.h>
#include <mma.h>

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include "common.cuh"

namespace {

constexpr int kWarpSize = 32;
constexpr int kHeadDim = 192;
constexpr int kQHeads = 8;
constexpr int kMma = 16;
constexpr int kKSteps = kHeadDim / kMma;
constexpr int kDimTilesPerWarp = kKSteps / 2;
constexpr int kCubeC = 64;
constexpr int kDimBlks = kHeadDim / kCubeC;
constexpr int kTilesPerBlk = kCubeC / kMma;
constexpr int kTokGrp = 8;

#ifndef AIU_MODS
  #define AIU_MODS "cs.padz"
#endif

#if STEP5_DSA_TC02
template <typename Frag>
__device__ __forceinline__ void ldm_x4(Frag& f, const __ppu_bfloat16* p) {
  unsigned* d = reinterpret_cast<unsigned*>(&f.x[0]);
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile(
      "ppu.tc02.ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, "
      "[%4];\n\t"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a));
}

template <typename Frag>
__device__ __forceinline__ void ldm_x4_trans(Frag& f, const __ppu_bfloat16* p) {
  unsigned* d = reinterpret_cast<unsigned*>(&f.x[0]);
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile(
      "ppu.tc02.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, "
      "%3}, [%4];\n\t"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a));
}

__device__ __forceinline__ void aiu_copy_group(__ppu_bfloat16* dst_shared,
                                               const __ppu_bfloat16* src_tensor,
                                               int dim_w, int start_w,
                                               int cube_w,
                                               int64_t stride_w_bytes) {
  const unsigned dst =
      static_cast<unsigned>(__cvta_generic_to_shared(dst_shared));
  asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile." AIU_MODS
               ".swzl.b16 "
               "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, "
               "%12}, %13;\n\t"
               :
               : "r"(dst), "l"(src_tensor), "r"(kCubeC), "r"(dim_w),
                 "r"(kDimBlks), "r"(kCubeC), "r"(cube_w), "r"(kDimBlks),
                 "l"(stride_w_bytes), "l"(static_cast<int64_t>(kCubeC) * 2),
                 "r"(0), "r"(start_w), "r"(0), "r"(0)
               : "memory");
}
#else
// ppu0010 build. The launcher refuses that core, so these stand-ins only have
// to keep the translation unit compilable and leave no uninitialised staging.
template <typename Frag>
__device__ __forceinline__ void ldm_x4(Frag& f, const __ppu_bfloat16*) {
  awmma::fill_fragment(f, __float2bfloat16(0.0f));
}

template <typename Frag>
__device__ __forceinline__ void ldm_x4_trans(Frag& f, const __ppu_bfloat16*) {
  awmma::fill_fragment(f, __float2bfloat16(0.0f));
}

__device__ __forceinline__ void aiu_copy_group(__ppu_bfloat16* dst_shared,
                                               const __ppu_bfloat16*, int, int,
                                               int, int64_t) {
  for (int i = threadIdx.x; i < kDimBlks * kTokGrp * kCubeC; i += 64)
    dst_shared[i] = __float2bfloat16(0.0f);
}
#endif

struct DsaPagedKvLayout {
  int page_size;
  int page_shift;
  int64_t page_stride;
  int64_t token_stride;
  int64_t head_stride;
};

__device__ __forceinline__ int step4_dsa_paged_decode_visible_count(
    int start, int64_t position) {
  if (start < 0 || position < start) return 0;
  position = position > 2147483647LL ? 2147483647LL : position;
  const int64_t distance = position - start;
  return distance >= 7 ? 8 : static_cast<int>(distance) + 1;
}

__device__ __forceinline__ void aiu_copy_paged_group(__ppu_bfloat16* dst,
                                                     const __ppu_bfloat16* src,
                                                     int physical_tokens,
                                                     int token,
                                                     DsaPagedKvLayout layout) {
  const bool valid =
      static_cast<unsigned>(token) < static_cast<unsigned>(physical_tokens);
  const int page = valid ? token >> layout.page_shift : 0;
  const int offset = valid ? token & (layout.page_size - 1) : layout.page_size;
  aiu_copy_group(dst, src + static_cast<int64_t>(page) * layout.page_stride,
                 layout.page_size, offset, 8, layout.token_stride * 2);
}

__device__ __forceinline__ float head_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}

__device__ __forceinline__ float head_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

__device__ __forceinline__ int frag_col(int t0, int i) {
  return 2 * t0 + (i & 1) + 8 * ((i >> 2) & 1);
}

__device__ __forceinline__ bool frag_is_real_row(int i) {
  return ((i >> 1) & 1) == 0;
}

__device__ __forceinline__ int frag_token(int i) { return (i >> 1) & 1; }

__device__ __forceinline__ int vmask_bit(int ti, int i) {
  return ti * 4 + (i & 1) + 2 * ((i >> 2) & 1);
}

__device__ __forceinline__ int vmask_bit2(int ti, int i) {
  return ti * 8 + ((i >> 1) & 1) * 4 + (i & 1) + 2 * ((i >> 2) & 1);
}

template <int HEAD_TILES, int SPLITS, bool FUSED_HEADS, int PAGED_TILE_S>
__device__ __forceinline__ void attention_body(
    const __ppu_bfloat16* __restrict__ query,
    const __ppu_bfloat16* __restrict__ key,
    const __ppu_bfloat16* __restrict__ value,
    const int32_t* __restrict__ region_ids,
    const int64_t* __restrict__ query_positions,
    __ppu_bfloat16* __restrict__ output, int64_t stride_qt, int64_t stride_qh,
    int64_t stride_rq, int64_t stride_rh, int64_t stride_ot, int64_t stride_oh,
    int kv_heads, int region_k, int kv_len, int q_len, float scale,
    DsaPagedKvLayout kv_layout, unsigned char* scratch, int program,
    const int32_t* packed_starts, float* split_max, float* split_denom,
    float* split_acc) {
  static_assert(
      SPLITS == 1 || SPLITS == 2 || SPLITS == 4 || SPLITS == 8 || SPLITS == 16,
      "unsupported split count");
  static_assert(HEAD_TILES == 1 || HEAD_TILES == 2, "expected GQA8 or GQA16");
  static_assert(!FUSED_HEADS || HEAD_TILES == 2, "16-head CTA requires GQA16");
  static_assert(PAGED_TILE_S == 32 || PAGED_TILE_S == 48,
                "staging tile must contain 32 or 48 tokens");
  using namespace awmma;
  constexpr int REGION_BLOCK = 8;
  constexpr bool kFullHeads = FUSED_HEADS;
  constexpr int kTokTiles = PAGED_TILE_S / kMma;
  constexpr int kTokGrps = PAGED_TILE_S / kTokGrp;
  constexpr int kRegionsPerTile = PAGED_TILE_S / REGION_BLOCK;
  using Stage = __ppu_bfloat16[kTokGrps][kDimBlks][kTokGrp][kCubeC];
  auto& sK = *reinterpret_cast<Stage*>(scratch);
  auto& sV = *reinterpret_cast<Stage*>(scratch + sizeof(Stage));

  const int q_token = program / (kv_heads * SPLITS);
  const int kv_head = (program / SPLITS) % kv_heads;
  const int split_id = program % SPLITS;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int t0 = lane & 3;
  const int head = lane >> 2;
  const int lr = lane & 7;
  const int lu = lane >> 3;
  const int a_col = 8 * (lu >> 1);
  const int b_col = 8 * (lu & 1);
  const int swzl_row = lr;
  const int64_t qpos = q_token < q_len ? query_positions[q_token] : -1;
  const int64_t rid_base = static_cast<int64_t>(q_token) * stride_rq +
                           static_cast<int64_t>(kv_head) * stride_rh;
  const int regions_per_split = (region_k + SPLITS - 1) / SPLITS;
  const int split_slot_base = split_id * regions_per_split;
  const int walk_len = regions_per_split;
  const bool paged_long_context =
      qpos >= static_cast<int64_t>(region_k) * REGION_BLOCK;

  fragment<accumulator, kMma, kMma, kMma, float> oacc[kDimTilesPerWarp];
#pragma unroll
  for (int i = 0; i < kDimTilesPerWarp; ++i) fill_fragment(oacc[i], 0.0f);
  const int dim0 = (warp % 2) * kDimTilesPerWarp * kMma;

  float running_max[2] = {-INFINITY, -INFINITY};
  float denom[2] = {0.0f, 0.0f};

  fragment<matrix_a, kMma, kMma, kMma, __ppu_bfloat16, row_major> qfrag[kKSteps];
  const __ppu_bfloat16* qrow =
      query + static_cast<int64_t>(q_token) * stride_qt +
      (static_cast<int64_t>(kv_head) * (kQHeads * HEAD_TILES) +
       (HEAD_TILES == 1
            ? 0
            : (FUSED_HEADS ? warp / 2 : static_cast<int>(blockIdx.z)) *
                  kQHeads) +
       head) *
          stride_qh;
#pragma unroll
  for (int k = 0; k < kKSteps; ++k) {
    fill_fragment(qfrag[k], __float2bfloat16(0.0f));
    *reinterpret_cast<__ppu_bfloat162*>(&qfrag[k].x[0]) =
        *reinterpret_cast<const __ppu_bfloat162*>(qrow + k * kMma + 2 * t0);
    *reinterpret_cast<__ppu_bfloat162*>(&qfrag[k].x[4]) =
        *reinterpret_cast<const __ppu_bfloat162*>(qrow + k * kMma + 2 * t0 + 8);
    if constexpr (kFullHeads) {
      const __ppu_bfloat16* qrow1 = qrow + 8 * stride_qh;
      *reinterpret_cast<__ppu_bfloat162*>(&qfrag[k].x[2]) =
          *reinterpret_cast<const __ppu_bfloat162*>(qrow1 + k * kMma + 2 * t0);
      *reinterpret_cast<__ppu_bfloat162*>(&qfrag[k].x[6]) =
          *reinterpret_cast<const __ppu_bfloat162*>(qrow1 + k * kMma + 2 * t0 +
                                                   8);
    }
  }

  constexpr int kVVecs = sizeof(sV) / sizeof(uint4);
  uint4* const vvec = reinterpret_cast<uint4*>(&sV[0][0][0][0]);
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
  for (int i = threadIdx.x; i < kVVecs; i += blockDim.x) vvec[i] = zero;
  __syncthreads();

  for (int slot0 = 0; slot0 < walk_len; slot0 += kRegionsPerTile) {
    int rids[kRegionsPerTile];
    int logical_starts[kRegionsPerTile];
#pragma unroll
    for (int rr = 0; rr < kRegionsPerTile; ++rr) {
      const int slot = split_slot_base + slot0 + rr;
      if (slot >= split_slot_base + walk_len || slot >= region_k) {
        rids[rr] = -1;
        logical_starts[rr] = -1;
      } else {
        rids[rr] = region_ids[rid_base + static_cast<int64_t>(slot)];
        logical_starts[rr] =
            packed_starts[rid_base + static_cast<int64_t>(slot)];
      }
    }

    bool tile_live = paged_long_context;
    if (!tile_live) {
#pragma unroll
      for (int rr = 0; rr < kRegionsPerTile; ++rr) {
        if (rids[rr] >= 0 && rids[rr] < kv_len / REGION_BLOCK &&
            logical_starts[rr] >= 0 && logical_starts[rr] <= qpos)
          tile_live = true;
      }
    }
    if (!tile_live) continue;

    {
      const __ppu_bfloat16* tensor =
          (warp == 0 ? key : value) +
          static_cast<int64_t>(kv_head) * kv_layout.head_stride;
      __ppu_bfloat16* dstbuf = warp == 0 ? &sK[0][0][0][0] : &sV[0][0][0][0];
      constexpr int kGrpElems = kDimBlks * kTokGrp * kCubeC;
#pragma unroll
      for (int tg = 0; tg < kTokGrps; ++tg) {
        const int r = rids[tg];
        const int start = r * REGION_BLOCK;
        if (r < 0 || r >= kv_len / REGION_BLOCK || logical_starts[tg] < 0 ||
            logical_starts[tg] > qpos)
          continue;
        aiu_copy_paged_group(dstbuf + tg * kGrpElems, tensor, kv_len, start,
                             kv_layout);
      }
      asm volatile("ppu.cp.async.commit_group;\n\t" ::: "memory");
    }

    unsigned vmask = 0;
#pragma unroll
    for (int ti = 0; ti < kTokTiles; ++ti) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (!kFullHeads && !frag_is_real_row(i)) continue;
        const int t = ti * kMma + frag_col(t0, i);
        const int r = rids[t / REGION_BLOCK];
        const int64_t tok =
            static_cast<int64_t>(logical_starts[t / REGION_BLOCK]) +
            (t % REGION_BLOCK);
        if (r >= 0 && r < kv_len / REGION_BLOCK &&
            logical_starts[t / REGION_BLOCK] >= 0 && tok <= qpos) {
          vmask |= 1u << (kFullHeads ? vmask_bit2(ti, i) : vmask_bit(ti, i));
        }
      }
    }

    asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
    __syncthreads();

    fragment<accumulator, kMma, kMma, kMma, float> sacc[kTokTiles];
    {
      fragment<matrix_b, kMma, kMma, kMma, __ppu_bfloat16, col_major> kfrag;
#pragma unroll
      for (int ti = 0; ti < kTokTiles; ++ti) fill_fragment(sacc[ti], 0.0f);
#pragma unroll
      for (int k = 0; k < kKSteps; ++k) {
        const int kdb = k / kTilesPerBlk;
        const int kchunk =
            (((k % kTilesPerBlk) * kMma + b_col) >> 3) ^ swzl_row;

#pragma unroll
        for (int ti = 0; ti < kTokTiles; ++ti) {
          ldm_x4(kfrag, &sK[ti * 2 + (lu >> 1)][kdb][lr][kchunk << 3]);
          mma_sync(sacc[ti], qfrag[k], kfrag, sacc[ti]);
        }
      }
    }

    if constexpr (kFullHeads) {
      float local_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
      for (int ti = 0; ti < kTokTiles; ++ti) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int tau = frag_token(i);
          const float score =
              (vmask & (1u << vmask_bit2(ti, i))) ? sacc[ti].x[i] : -INFINITY;
          local_max[tau] = fmaxf(local_max[tau], score);
          sacc[ti].x[i] = score;
        }
      }
      float new_max[2], rescale[2];
#pragma unroll
      for (int tau = 0; tau < 2; ++tau) {
        const float tile_max = head_max(local_max[tau]);
        const bool has_valid = tile_max > -INFINITY;
        new_max[tau] =
            has_valid ? fmaxf(running_max[tau], tile_max) : running_max[tau];
        rescale[tau] = has_valid
                           ? __expf((running_max[tau] - new_max[tau]) * scale)
                           : 1.0f;
      }

      fragment<matrix_a, kMma, kMma, kMma, __ppu_bfloat16, row_major>
          pfrag[kTokTiles];
      float local_sum[2] = {0.0f, 0.0f};
#pragma unroll
      for (int ti = 0; ti < kTokTiles; ++ti) {
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
          const int tau = frag_token(i);
          const float offset = new_max[tau] == -INFINITY ? 0.0f : new_max[tau];
          float x0 = (sacc[ti].x[i] - offset) * scale;
          float x1 = (sacc[ti].x[i + 1] - offset) * scale;
          if (!(scale > 0.0f)) {
            if (!(vmask & (1u << vmask_bit2(ti, i)))) x0 = -INFINITY;
            if (!(vmask & (1u << vmask_bit2(ti, i + 1)))) x1 = -INFINITY;
          }
          const float p0 = __expf(x0), p1 = __expf(x1);
          const __ppu_bfloat162 pb = __floats2bfloat162_rn(p0, p1);
          *reinterpret_cast<__ppu_bfloat162*>(&pfrag[ti].x[i]) = pb;
          local_sum[tau] += p0;
          local_sum[tau] += p1;
        }
      }
#pragma unroll
      for (int tau = 0; tau < 2; ++tau) {
        denom[tau] = denom[tau] * rescale[tau] + local_sum[tau];
        running_max[tau] = new_max[tau];
      }

#pragma unroll
      for (int j = 0; j < kDimTilesPerWarp; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i) oacc[j].x[i] *= rescale[frag_token(i)];
      }

      if (warp == 1)
        asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
      __syncthreads();

      {
        fragment<matrix_b, kMma, kMma, kMma, __ppu_bfloat16, row_major> vfrag;
#pragma unroll
        for (int ti = 0; ti < kTokTiles; ++ti) {
#pragma unroll
          for (int j = 0; j < kDimTilesPerWarp; ++j) {
            const int vdim = dim0 + j * kMma;
            const int vdb = vdim / kCubeC;
            const int vchunk = (((vdim % kCubeC) + a_col) >> 3) ^ swzl_row;
            ldm_x4_trans(vfrag, &sV[ti * 2 + (lu & 1)][vdb][lr][vchunk << 3]);
            mma_sync(oacc[j], pfrag[ti], vfrag, oacc[j]);
          }
        }
      }
      __syncthreads();
      continue;
    }

    float local_max = -INFINITY;
#pragma unroll
    for (int ti = 0; ti < kTokTiles; ++ti) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (!frag_is_real_row(i)) continue;
        if (vmask & (1u << vmask_bit(ti, i))) {
          local_max = fmaxf(local_max, sacc[ti].x[i]);
        }
      }
    }
    const float tile_max = head_max(local_max);
    const bool has_valid = tile_max > -INFINITY;
    const float new_max =
        has_valid ? fmaxf(running_max[0], tile_max) : running_max[0];
    const float rescale =
        has_valid ? __expf((running_max[0] - new_max) * scale) : 1.0f;

    fragment<matrix_a, kMma, kMma, kMma, __ppu_bfloat16, row_major>
        pfrag[kTokTiles];
    float local_sum = 0.0f;
#pragma unroll
    for (int ti = 0; ti < kTokTiles; ++ti) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float p = 0.0f;
        if (frag_is_real_row(i) && (vmask & (1u << vmask_bit(ti, i)))) {
          p = __expf((sacc[ti].x[i] - new_max) * scale);
        }
        const __ppu_bfloat16 pb = __float2bfloat16(p);
        pfrag[ti].x[i] = pb;
        local_sum += p;
      }
    }
    denom[0] = denom[0] * rescale + local_sum;
    running_max[0] = new_max;

#pragma unroll
    for (int j = 0; j < kDimTilesPerWarp; ++j) {
#pragma unroll
      for (int i = 0; i < 8; ++i) oacc[j].x[i] *= rescale;
    }

    {
      fragment<matrix_b, kMma, kMma, kMma, __ppu_bfloat16, row_major> vfrag;
#pragma unroll
      for (int ti = 0; ti < kTokTiles; ++ti) {
#pragma unroll
        for (int j = 0; j < kDimTilesPerWarp; ++j) {
          const int vdim = dim0 + j * kMma;
          const int vdb = vdim / kCubeC;
          const int vchunk = (((vdim % kCubeC) + a_col) >> 3) ^ swzl_row;
          ldm_x4_trans(vfrag, &sV[ti * 2 + (lu & 1)][vdb][lr][vchunk << 3]);
          mma_sync(oacc[j], pfrag[ti], vfrag, oacc[j]);
        }
      }
    }
    __syncthreads();
  }

  if constexpr (kFullHeads) {
#pragma unroll
    for (int tau = 0; tau < 2; ++tau) {
      const float d = head_sum(denom[tau]);
      const int64_t gh = static_cast<int64_t>(kv_head) * 16 + head + tau * 8;
      const int64_t scalar =
          (static_cast<int64_t>(q_token) * kv_heads * 16 + gh) * SPLITS +
          split_id;
      if (warp == 0 && t0 == 0) {
        split_max[scalar] = d > 0.0f ? running_max[tau] * scale : -INFINITY;
        split_denom[scalar] = d;
      }
#pragma unroll
      for (int j = 0; j < kDimTilesPerWarp; ++j) {
        const int64_t base = scalar * kHeadDim + dim0 + j * kMma;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int i0 = 2 * tau + 4 * half;
          split_acc[base + 2 * t0 + 8 * half] = oacc[j].x[i0];
          split_acc[base + 2 * t0 + 8 * half + 1] = oacc[j].x[i0 + 1];
        }
      }
    }
  } else {
    const float d = head_sum(denom[0]);
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
    const int64_t gh =
        static_cast<int64_t>(kv_head) * (kQHeads * HEAD_TILES) +
        (HEAD_TILES == 1
             ? 0
             : (FUSED_HEADS ? warp / 2 : static_cast<int>(blockIdx.z)) *
                   kQHeads) +
        head;
    const int64_t o_base =
        static_cast<int64_t>(q_token) * stride_ot + gh * stride_oh;
    if constexpr (SPLITS > 1) {
      const int64_t scalar =
          (static_cast<int64_t>(q_token) * kv_heads * kQHeads * HEAD_TILES +
           gh) *
              SPLITS +
          split_id;
      if ((FUSED_HEADS ? (warp % 2 == 0) : (warp == 0)) && t0 == 0) {
        split_max[scalar] = d > 0.0f ? running_max[0] * scale : -INFINITY;
        split_denom[scalar] = d;
      }
      const int64_t acc_base = scalar * kHeadDim;
#pragma unroll
      for (int j = 0; j < kDimTilesPerWarp; ++j) {
        const int64_t base = acc_base + dim0 + j * kMma;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int i0 = half * 4;
          split_acc[base + 2 * t0 + 8 * half] = oacc[j].x[i0];
          split_acc[base + 2 * t0 + 8 * half + 1] = oacc[j].x[i0 + 1];
        }
      }
    } else {
      // A single split writes the output directly, but SGLang still needs the
      // natural LSE, so publish the (max, denominator) state it reduces from.
      const int64_t scalar =
          static_cast<int64_t>(q_token) * kv_heads * kQHeads * HEAD_TILES + gh;
      if ((FUSED_HEADS ? (warp % 2 == 0) : (warp == 0)) && t0 == 0) {
        split_max[scalar] = d > 0.0f ? running_max[0] * scale : -INFINITY;
        split_denom[scalar] = d;
      }
#pragma unroll
      for (int j = 0; j < kDimTilesPerWarp; ++j) {
        const int64_t base = o_base + dim0 + j * kMma;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int i0 = half * 4;
          const __ppu_bfloat162 pair = __floats2bfloat162_rn(
              oacc[j].x[i0] * inv, oacc[j].x[i0 + 1] * inv);
          *reinterpret_cast<__ppu_bfloat162*>(output + base + 2 * t0 +
                                             8 * half) = pair;
        }
      }
    }
  }
}

template <int SPLITS, int TILE>
__global__ __launch_bounds__(64, 2) void step4_dsa_paged_decode_ws_kernel(
    const __ppu_bfloat16* query, const __ppu_bfloat16* key,
    const __ppu_bfloat16* value, const int32_t* regions, const int32_t* starts,
    const int64_t* positions, float* maxima, float* denominators,
    float* accumulators, int kv_heads, int region_k, int physical_tokens,
    float scale, DsaPagedKvLayout kv_layout) {
  using namespace awmma;
  constexpr int R = TILE / 8;
  constexpr int NT = TILE / 16;
  using Stage = __ppu_bfloat16[R][3][8][64];
  __shared__ __align__(128) Stage sk;
  __shared__ __align__(128) Stage sv;
  __shared__ uint32_t probability[2][NT][4][32];
  __shared__ float factors[2][2][8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int head = lane >> 2;
  const int t0 = lane & 3;
  const int lr = lane & 7;
  const int lu = lane >> 3;
  const int qt = blockIdx.x / (kv_heads * SPLITS);
  const int kh = (blockIdx.x / SPLITS) % kv_heads;
  const int split = blockIdx.x % SPLITS;
  const int step = region_k >= SPLITS * R ? region_k / (SPLITS * R) * R
                                          : (region_k + SPLITS - 1) / SPLITS;
  const int begin = split * step;
  const int count = split == SPLITS - 1 ? region_k - begin : step;
  const int64_t qpos = positions[qt];
  const int64_t rb = (static_cast<int64_t>(qt) * kv_heads + kh) * region_k;
  for (int i = threadIdx.x; i < sizeof(Stage) / sizeof(uint4); i += 64)
    reinterpret_cast<uint4*>(&sv)[i] = make_uint4(0, 0, 0, 0);
  __syncthreads();
  if (warp == 0) {
    fragment<matrix_a, 16, 16, 16, __ppu_bfloat16, row_major> qf[12];
    const auto* q0 =
        query +
        (static_cast<int64_t>(qt) * kv_heads * 16 + kh * 16 + head) * 192;
#pragma unroll
    for (int k = 0; k < 12; ++k) {
      *reinterpret_cast<__ppu_bfloat162*>(&qf[k].x[0]) =
          *reinterpret_cast<const __ppu_bfloat162*>(q0 + k * 16 + 2 * t0);
      *reinterpret_cast<__ppu_bfloat162*>(&qf[k].x[4]) =
          *reinterpret_cast<const __ppu_bfloat162*>(q0 + k * 16 + 2 * t0 + 8);
      *reinterpret_cast<__ppu_bfloat162*>(&qf[k].x[2]) =
          *reinterpret_cast<const __ppu_bfloat162*>(q0 + 8 * 192 + k * 16 +
                                                   2 * t0);
      *reinterpret_cast<__ppu_bfloat162*>(&qf[k].x[6]) =
          *reinterpret_cast<const __ppu_bfloat162*>(q0 + 8 * 192 + k * 16 +
                                                   2 * t0 + 8);
    }
    float maximum[2] = {-INFINITY, -INFINITY};
    float denom[2] = {0, 0};
    int cr = -1, cs = -1;
    for (int b = 0; b < count; b += R) {
      if ((b & 31) == 0) {
        const int slot = begin + b + lane;
        const bool valid = slot < region_k && b + lane < count;
        cr = valid ? regions[rb + slot] : -1;
        cs = valid ? starts[rb + slot] : -1;
      }
      int ids[R], logical[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        ids[r] = __shfl_sync(0xffffffffu, cr, (b + r) & 31);
        logical[r] = __shfl_sync(0xffffffffu, cs, (b + r) & 31);
        if (b + r >= count || begin + b + r >= region_k) ids[r] = -1;
        if (ids[r] >= 0 && ids[r] < physical_tokens / 8 && logical[r] >= 0 &&
            logical[r] <= qpos)
          aiu_copy_paged_group(&sk[r][0][0][0],
                               key + kh * kv_layout.head_stride,
                               physical_tokens, ids[r] * 8, kv_layout);
      }
      asm volatile("ppu.cp.async.commit_group;\n\t" ::: "memory");
      asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
      fragment<accumulator, 16, 16, 16, float> scores[NT];
      fragment<matrix_b, 16, 16, 16, __ppu_bfloat16, col_major> kf;
#pragma unroll
      for (int t = 0; t < NT; ++t) fill_fragment(scores[t], 0.0f);
#pragma unroll
      for (int k = 0; k < 12; ++k) {
        const int chunk = (((k % 4) * 16 + 8 * (lu & 1)) >> 3) ^ lr;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          ldm_x4(kf, &sk[t * 2 + (lu >> 1)][k / 4][lr][chunk * 8]);
          mma_sync(scores[t], qf[k], kf, scores[t]);
        }
      }
      float lm[2] = {-INFINITY, -INFINITY};
      uint32_t valid_bits = 0;
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int token = t * 16 + frag_col(t0, i);
          const int r = token / 8;
          const int64_t pos = static_cast<int64_t>(logical[r]) + token % 8;
          const bool valid = ids[r] >= 0 && ids[r] < physical_tokens / 8 &&
                             logical[r] >= 0 && pos <= qpos;
          if (valid) valid_bits |= 1u << (t * 8 + i);
          const float s = valid ? scores[t].x[i] : -INFINITY;
          scores[t].x[i] = s;
          lm[frag_token(i)] = fmaxf(lm[frag_token(i)], s);
        }
      }
      float next[2], rescale[2];
#pragma unroll
      for (int tau = 0; tau < 2; ++tau) {
        const float tm = head_max(lm[tau]);
        const bool valid = tm > -INFINITY;
        next[tau] = valid ? fmaxf(maximum[tau], tm) : maximum[tau];
        rescale[tau] =
            valid ? __expf((maximum[tau] - next[tau]) * scale) : 1.0f;
      }
      const int stage = (b / R) & 1;
      float sum[2] = {0, 0};
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
          const int tau = frag_token(i);
          const float offset = next[tau] == -INFINITY ? 0.0f : next[tau];
          float x0 = (scores[t].x[i] - offset) * scale;
          float x1 = (scores[t].x[i + 1] - offset) * scale;
          if (!(scale > 0)) {
            if (!(valid_bits & (1u << (t * 8 + i)))) x0 = -INFINITY;
            if (!(valid_bits & (1u << (t * 8 + i + 1)))) x1 = -INFINITY;
          }
          const float p0 = __expf(x0), p1 = __expf(x1);
          const __ppu_bfloat162 packed = __floats2bfloat162_rn(p0, p1);
          probability[stage][t][i / 2][lane] =
              *reinterpret_cast<const uint32_t*>(&packed);
          sum[tau] += p0;
          sum[tau] += p1;
        }
      }
#pragma unroll
      for (int tau = 0; tau < 2; ++tau) {
        denom[tau] = denom[tau] * rescale[tau] + sum[tau];
        maximum[tau] = next[tau];
        if (t0 == 0) factors[stage][tau][head] = rescale[tau];
      }
      __ppu_barrier_sync(1, 64);
    }
#pragma unroll
    for (int tau = 0; tau < 2; ++tau) {
      const float d = head_sum(denom[tau]);
      const int64_t scalar =
          ((static_cast<int64_t>(qt) * kv_heads + kh) * 16 + head + tau * 8) *
              SPLITS +
          split;
      if (t0 == 0) {
        maxima[scalar] = d > 0 ? maximum[tau] * scale : -INFINITY;
        denominators[scalar] = d;
      }
    }
  } else {
    fragment<accumulator, 16, 16, 16, float> out[12];
#pragma unroll
    for (int j = 0; j < 12; ++j) fill_fragment(out[j], 0.0f);
    int cr = -1, cs = -1;
    for (int b = 0; b < count; b += R) {
      if ((b & 31) == 0) {
        const int slot = begin + b + lane;
        const bool valid = slot < region_k && b + lane < count;
        cr = valid ? regions[rb + slot] : -1;
        cs = valid ? starts[rb + slot] : -1;
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const int id = __shfl_sync(0xffffffffu, cr, (b + r) & 31);
        const int start = __shfl_sync(0xffffffffu, cs, (b + r) & 31);
        if (b + r < count && begin + b + r < region_k && id >= 0 &&
            id < physical_tokens / 8 && start >= 0 && start <= qpos)
          aiu_copy_paged_group(&sv[r][0][0][0],
                               value + kh * kv_layout.head_stride,
                               physical_tokens, id * 8, kv_layout);
      }
      asm volatile("ppu.cp.async.commit_group;\n\t" ::: "memory");
      asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
      __ppu_barrier_sync(1, 64);
      const int stage = (b / R) & 1;
      const float r0 = factors[stage][0][head];
      const float r1 = factors[stage][1][head];
#pragma unroll
      for (int j = 0; j < 12; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i) out[j].x[i] *= frag_token(i) == 0 ? r0 : r1;
      }
      fragment<matrix_a, 16, 16, 16, __ppu_bfloat16, row_major> pf;
      fragment<matrix_b, 16, 16, 16, __ppu_bfloat16, row_major> vf;
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
          reinterpret_cast<uint32_t*>(&pf.x[0])[i] =
              probability[stage][t][i][lane];
#pragma unroll
        for (int j = 0; j < 12; ++j) {
          const int chunk = (((j % 4) * 16 + 8 * (lu >> 1)) >> 3) ^ lr;
          ldm_x4_trans(vf, &sv[t * 2 + (lu & 1)][j / 4][lr][chunk * 8]);
          mma_sync(out[j], pf, vf, out[j]);
        }
      }
    }
#pragma unroll
    for (int tau = 0; tau < 2; ++tau) {
      const int64_t scalar =
          ((static_cast<int64_t>(qt) * kv_heads + kh) * 16 + head + tau * 8) *
              SPLITS +
          split;
#pragma unroll
      for (int j = 0; j < 12; ++j) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int i = tau * 2 + half * 4;
          const int64_t dst = scalar * 192 + j * 16 + 2 * t0 + half * 8;
          accumulators[dst] = out[j].x[i];
          accumulators[dst + 1] = out[j].x[i + 1];
        }
      }
    }
  }
}
template <int SPLITS, int WARPS>
__global__ void step4_dsa_paged_decode_merge_parallel(const float* maxima,
                                                      const float* denominators,
                                                      const float* accumulators,
                                                      __ppu_bfloat16* output) {
  const int head = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  __shared__ float weights[SPLITS];
  __shared__ float sums[WARPS][192];
  __shared__ float inv;
  if (warp == 0) {
    float m = -INFINITY;
#pragma unroll
    for (int s = lane; s < SPLITS; s += 32)
      if (denominators[head * SPLITS + s] > 0)
        m = fmaxf(m, maxima[head * SPLITS + s]);
#pragma unroll
    for (int d = 16; d > 0; d >>= 1)
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));
    float sum = 0;
#pragma unroll
    for (int s = lane; s < SPLITS; s += 32) {
      const float den = denominators[head * SPLITS + s];
      const float e = den > 0 ? __expf(maxima[head * SPLITS + s] - m) : 0.0f;
      weights[s] = e;
      sum += e * den;
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1)
      sum += __shfl_xor_sync(0xffffffffu, sum, d);
    if (lane == 0) inv = sum > 0 ? 1.0f / sum : 0.0f;
  }
  __syncthreads();
  float acc[6] = {};
#pragma unroll
  for (int s = warp; s < SPLITS; s += WARPS) {
    const float factor = weights[s];
#pragma unroll
    for (int j = 0; j < 6; ++j)
      acc[j] += accumulators[(static_cast<int64_t>(head) * SPLITS + s) * 192 +
                             lane + j * 32] *
                factor;
  }
#pragma unroll
  for (int j = 0; j < 6; ++j) sums[warp][lane + j * 32] = acc[j];
  __syncthreads();
  for (int d = threadIdx.x; d < 192; d += WARPS * 32) {
    float total = 0;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) total += sums[w][d];
    output[head * 192 + d] = __float2bfloat16(total * inv);
  }
}

template <int SPLITS>
__global__ void step4_dsa_paged_decode_merge_native(const float* split_max,
                                                    const float* split_denom,
                                                    const float* split_acc,
                                                    __ppu_bfloat16* output) {
  const int qt = blockIdx.x / 12, j = blockIdx.x % 12;
  const int tid = threadIdx.x & 63;
  const int tau = tid / 32, half = (tid & 31) / 16;
  const int head = qt * 16 + (tid & 15) / 2 + tau * 8;
  const int col = j * 16 + half * 8 + (tid & 1) * 4;
  float ms[SPLITS], ds[SPLITS];
  float4 xs[SPLITS];
#pragma unroll
  for (int s = 0; s < SPLITS; ++s) {
    ds[s] = split_denom[head * SPLITS + s];
    ms[s] = ds[s] > 0 ? split_max[head * SPLITS + s] : -INFINITY;
    xs[s] = ds[s] > 0 ? __ldcg(reinterpret_cast<const float4*>(split_acc) +
                               ((qt * SPLITS + s) * 12 + j) * 64 + tid)
                      : make_float4(0, 0, 0, 0);
  }
  float m = -INFINITY, denom = 0.0f;
#pragma unroll
  for (int s = 0; s < SPLITS; ++s) m = fmaxf(m, ms[s]);
  float4 acc = make_float4(0, 0, 0, 0);
#pragma unroll
  for (int s = 0; s < SPLITS; ++s) {
    const float weight = ds[s] > 0 ? __expf(ms[s] - m) : 0.0f;
    denom += ds[s] * weight;
    acc.x += xs[s].x * weight;
    acc.y += xs[s].y * weight;
    acc.z += xs[s].z * weight;
    acc.w += xs[s].w * weight;
  }
  const float inv = denom > 0 ? 1.0f / denom : 0.0f;
  auto* dst = reinterpret_cast<__ppu_bfloat162*>(output + head * 192 + col);
  dst[0] = __floats2bfloat162_rn(acc.x * inv, acc.y * inv);
  dst[1] = __floats2bfloat162_rn(acc.z * inv, acc.w * inv);
}
template <int SPLITS>
__global__ __launch_bounds__(32, 2) void step4_dsa_paged_decode_compact_kernel(
    const __ppu_bfloat16* query, const __ppu_bfloat16* key,
    const __ppu_bfloat16* value, const int32_t* regions, const int32_t* starts,
    const int64_t* positions, float* maxima, float* denominators,
    float* accumulators, int kv_heads, int region_k, int physical_tokens,
    float scale, DsaPagedKvLayout kv_layout) {
  using namespace awmma;
  constexpr int R = 2;
  constexpr int NT = 1;
  using Stage = __ppu_bfloat16[R][3][8][64];
  __shared__ __align__(128) Stage sk;
  Stage& sv = sk;
  const int lane = threadIdx.x & 31;
  const int head = lane >> 2;
  const int t0 = lane & 3;
  const int lr = lane & 7;
  const int lu = lane >> 3;
  const int qt = blockIdx.x / (kv_heads * SPLITS);
  const int kh = (blockIdx.x / SPLITS) % kv_heads;
  const int split = blockIdx.x % SPLITS;
  const int step = region_k >= SPLITS * R ? region_k / (SPLITS * R) * R
                                          : (region_k + SPLITS - 1) / SPLITS;
  const int begin = split * step;
  const int count = split == SPLITS - 1 ? region_k - begin : step;
  const int64_t qpos = positions[qt];
  const int64_t rb = (static_cast<int64_t>(qt) * kv_heads + kh) * region_k;
  fragment<matrix_a, 16, 16, 16, __ppu_bfloat16, row_major> qf[12];
  const auto* qp =
      query + (static_cast<int64_t>(qt) * kv_heads * 16 + kh * 16) * 192;
#pragma unroll
  for (int t = 0; t < 2; ++t)
    aiu_copy_group(&sk[t][0][0][0], qp, 16, t * 8, 8, 384);
  asm volatile("ppu.cp.async.commit_group;\n\t" ::: "memory");
  asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
#pragma unroll
  for (int k = 0; k < 12; ++k) {
    const int qc = (((k % 4) * 16 + 8 * (lu >> 1)) >> 3) ^ lr;
    ldm_x4(qf[k], &sk[lu & 1][k / 4][lr][qc * 8]);
  }
  fragment<accumulator, 16, 16, 16, float> out[12];
#pragma unroll
  for (int j = 0; j < 12; ++j) fill_fragment(out[j], 0.0f);
  float maximum[2] = {-INFINITY, -INFINITY};
  float denom[2] = {0, 0};
  int cr = -1, cs = -1;
  for (int b = 0; b < count; b += R) {
    if ((b & 31) == 0) {
      const int slot = begin + b + lane;
      const bool valid = slot < region_k && b + lane < count;
      cr = valid ? regions[rb + slot] : -1;
      cs = valid ? starts[rb + slot] : -1;
    }
    int ids[R], logical[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      ids[r] = __shfl_sync(0xffffffffu, cr, (b + r) & 31);
      logical[r] = __shfl_sync(0xffffffffu, cs, (b + r) & 31);
      if (b + r >= count || begin + b + r >= region_k) ids[r] = -1;
      if (ids[r] >= 0 && ids[r] < physical_tokens / 8 && logical[r] >= 0 &&
          logical[r] <= qpos)
        aiu_copy_paged_group(&sk[r][0][0][0], key + kh * kv_layout.head_stride,
                             physical_tokens, ids[r] * 8, kv_layout);
    }
    asm volatile("ppu.cp.async.commit_group;\n\t" ::: "memory");
    asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
    fragment<accumulator, 16, 16, 16, float> scores[NT];
    fragment<matrix_b, 16, 16, 16, __ppu_bfloat16, col_major> kf;
#pragma unroll
    for (int t = 0; t < NT; ++t) fill_fragment(scores[t], 0.0f);
#pragma unroll
    for (int k = 0; k < 12; ++k) {
      const int chunk = (((k % 4) * 16 + 8 * (lu & 1)) >> 3) ^ lr;
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        ldm_x4(kf, &sk[t * 2 + (lu >> 1)][k / 4][lr][chunk * 8]);
        mma_sync(scores[t], qf[k], kf, scores[t]);
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const bool valid = ids[r] >= 0 && ids[r] < physical_tokens / 8 &&
                         logical[r] >= 0 && logical[r] <= qpos;
      aiu_copy_paged_group(&sv[r][0][0][0], value + kh * kv_layout.head_stride,
                           physical_tokens,
                           valid ? ids[r] * 8 : physical_tokens, kv_layout);
    }
    asm volatile("ppu.cp.async.commit_group;\n\t" ::: "memory");
    float lm[2] = {-INFINITY, -INFINITY};
    uint32_t valid_bits = 0;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int token = t * 16 + frag_col(t0, i);
        const int r = token / 8;
        const int64_t pos = static_cast<int64_t>(logical[r]) + token % 8;
        const bool valid = ids[r] >= 0 && ids[r] < physical_tokens / 8 &&
                           logical[r] >= 0 && pos <= qpos;
        if (valid) valid_bits |= 1u << (t * 8 + i);
        const float s = valid ? scores[t].x[i] : -INFINITY;
        scores[t].x[i] = s;
        lm[frag_token(i)] = fmaxf(lm[frag_token(i)], s);
      }
    }
    float next[2], rescale[2];
#pragma unroll
    for (int tau = 0; tau < 2; ++tau) {
      const float tm = head_max(lm[tau]);
      const bool valid = tm > -INFINITY;
      next[tau] = valid ? fmaxf(maximum[tau], tm) : maximum[tau];
      rescale[tau] = valid ? __expf((maximum[tau] - next[tau]) * scale) : 1.0f;
    }
    fragment<matrix_a, 16, 16, 16, __ppu_bfloat16, row_major> pf[NT];
    float sum[2] = {0, 0};
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const int tau = frag_token(i);
        const float offset = next[tau] == -INFINITY ? 0.0f : next[tau];
        float x0 = (scores[t].x[i] - offset) * scale;
        float x1 = (scores[t].x[i + 1] - offset) * scale;
        if (!(scale > 0)) {
          if (!(valid_bits & (1u << (t * 8 + i)))) x0 = -INFINITY;
          if (!(valid_bits & (1u << (t * 8 + i + 1)))) x1 = -INFINITY;
        }
        const float p0 = __expf(x0), p1 = __expf(x1);
        const __ppu_bfloat162 packed = __floats2bfloat162_rn(p0, p1);
        reinterpret_cast<uint32_t*>(&pf[t].x[0])[i / 2] =
            *reinterpret_cast<const uint32_t*>(&packed);
        sum[tau] += p0;
        sum[tau] += p1;
      }
    }
#pragma unroll
    for (int tau = 0; tau < 2; ++tau) {
      denom[tau] = denom[tau] * rescale[tau] + sum[tau];
      maximum[tau] = next[tau];
    }
    asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
#pragma unroll
    for (int j = 0; j < 12; ++j) {
#pragma unroll
      for (int i = 0; i < 8; ++i)
        out[j].x[i] *= frag_token(i) == 0 ? rescale[0] : rescale[1];
    }
    fragment<matrix_b, 16, 16, 16, __ppu_bfloat16, row_major> vf;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int j = 0; j < 12; ++j) {
        const int chunk = (((j % 4) * 16 + 8 * (lu >> 1)) >> 3) ^ lr;
        ldm_x4_trans(vf, &sv[t * 2 + (lu & 1)][j / 4][lr][chunk * 8]);
        mma_sync(out[j], pf[t], vf, out[j]);
      }
    }
  }
#pragma unroll
  for (int tau = 0; tau < 2; ++tau) {
    const float d = head_sum(denom[tau]);
    const int64_t scalar =
        ((static_cast<int64_t>(qt) * kv_heads + kh) * 16 + head + tau * 8) *
            SPLITS +
        split;
    if (t0 == 0) {
      maxima[scalar] = d > 0 ? maximum[tau] * scale : -INFINITY;
      denominators[scalar] = d;
    }
#pragma unroll
    for (int j = 0; j < 12; ++j) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int i = tau * 2 + half * 4;
        const int64_t dst = scalar * 192 + j * 16 + 2 * t0 + half * 8;
        accumulators[dst] = out[j].x[i];
        accumulators[dst + 1] = out[j].x[i + 1];
      }
    }
  }
}
constexpr int kDsaPairMetadataThreads = 1024;

__device__ __forceinline__ int step4_dsa_paged_decode_pair_prefix(int x,
                                                                  int* sums,
                                                                  int& total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int prefix = x;
#pragma unroll
  for (int d = 1; d < 32; d *= 2) {
    const int v = __shfl_up_sync(0xffffffffu, prefix, d);
    if (lane >= d) prefix += v;
  }
  if (lane == 31) sums[warp] = prefix;
  __syncthreads();
  int warp_prefix = sums[lane];
#pragma unroll
  for (int d = 1; d < 32; d *= 2) {
    const int value = __shfl_up_sync(0xffffffffu, warp_prefix, d);
    if (lane >= d) warp_prefix += value;
  }
  const int before =
      warp == 0 ? 0 : __shfl_sync(0xffffffffu, warp_prefix, warp - 1);
  total = __shfl_sync(0xffffffffu, warp_prefix, 31);
  return before + prefix - x;
}

template <bool PACKED>
__global__ void step4_dsa_paged_decode_pair_metadata(
    const int32_t* regions, const int32_t* starts, const int64_t* positions,
    int32_t* packed, int32_t* counts, int physical_regions) {
  __shared__ __align__(16) int keys[4096], slots[PACKED ? 1 : 4096];
  __shared__ int common[1024], sums[32];
  constexpr int n = 513, cap = 1026;
  const int pair = blockIdx.x, tid = threadIdx.x;
  const int64_t pos0 = positions[(pair / 2) * 4 + (pair & 1)];
  const int64_t pos1 = positions[(pair / 2) * 4 + (pair & 1) + 2];
  int unmatched = 0;
  reinterpret_cast<int4*>(keys)[tid] = make_int4(-1, -1, -1, -1);
  if constexpr (!PACKED)
    reinterpret_cast<int4*>(slots)[tid] = make_int4(-1, -1, -1, -1);
  common[tid] = 0;
  __syncthreads();
  if (tid < n) {
    const int i = tid;
    const int id = regions[((pair / 2) * 4 + (pair & 1)) * n + i];

    if (id >= 0 && id < physical_regions) {
      unsigned pos = (static_cast<unsigned>(id) * 2654435761u) >> 20;
      if constexpr (PACKED) {
        while (atomicCAS(keys + pos, -1, id | (i << 21)) != -1)
          pos = (pos + 1) & 4095;
      } else {
        while (atomicCAS(keys + pos, -1, id) != -1) pos = (pos + 1) & 4095;
        slots[pos] = i;
      }
    }
  }
  __syncthreads();
  if (tid < n) {
    const int i = tid;
    const int id = regions[((pair / 2) * 4 + (pair & 1) + 2) * n + i];
    const int logical = step4_dsa_paged_decode_visible_count(
        starts[((pair / 2) * 4 + (pair & 1) + 2) * n + i], pos1);
    int matched = -1;
    if (id >= 0 && id < physical_regions) {
      unsigned pos = (static_cast<unsigned>(id) * 2654435761u) >> 20;
      for (int probe = 0; probe < 4096; ++probe) {
        const int k = keys[pos];
        if (k == -1) break;
        if constexpr (PACKED) {
          if ((k & 2097151) == id && k >= 0) {
            const int old = atomicOr(keys + pos, static_cast<int>(0x80000000u));
            if (old >= 0) {
              matched = old >> 21;
              break;
            }
          }
        } else {
          if (k == id) {
            const int index = atomicExch(slots + pos, -1);
            if (index >= 0) {
              matched = index;
              break;
            }
          }
        }
        pos = (pos + 1) & 4095;
      }
    }
    unmatched = matched < 0 ? 1 : 0;
    if (matched >= 0) {
      common[matched] = 16 | logical;
    }
  }
  __syncthreads();
  const int x = (common[tid] != 0) + (unmatched << 16);
  int total = 0;
  const int prefix = step4_dsa_paged_decode_pair_prefix(x, sums, total);
  const int shared_count = total & 65535;
  const int common_prefix = prefix & 65535;
  const int unmatched_prefix = prefix >> 16;
  if (tid == 0) {
    counts[pair * 2 + 1] = shared_count;
    counts[pair * 2] = n + (total >> 16);
  }
  if (tid < n) {
    const int dst =
        common[tid] ? common_prefix : shared_count + tid - common_prefix;
    packed[(pair * 3) * cap + dst] =
        regions[((pair / 2) * 4 + (pair & 1)) * n + tid];
    packed[(pair * 3 + 1) * cap + dst] = step4_dsa_paged_decode_visible_count(
        starts[((pair / 2) * 4 + (pair & 1)) * n + tid], pos0);
    packed[(pair * 3 + 2) * cap + dst] = common[tid] & 15;
    if (unmatched) {
      const int dst1 = n + unmatched_prefix;
      packed[(pair * 3) * cap + dst1] =
          regions[((pair / 2) * 4 + (pair & 1) + 2) * n + tid];
      packed[(pair * 3 + 1) * cap + dst1] = -1;
      packed[(pair * 3 + 2) * cap + dst1] =
          step4_dsa_paged_decode_visible_count(
              starts[((pair / 2) * 4 + (pair & 1) + 2) * n + tid], pos1);
    }
  }
}

__device__ __forceinline__ void step4_dsa_paged_decode_copy_group(
    __ppu_bfloat16* dst_shared, const __ppu_bfloat16* src_tensor, int dim_w,
    int start_w, int cube_w, int64_t stride_w_bytes) {
#if STEP5_DSA_TC02
  const unsigned dst =
      static_cast<unsigned>(__cvta_generic_to_shared(dst_shared));
  asm volatile(
      "ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.rt.padz"
      ".swzl.b16 "
      "[%0], [%1], {%2, %3, %4}, {%5, %6, %7}, {%8, %9}, {%10, %11, %12}, "
      "%13;\n\t"
      :
      : "r"(dst), "l"(src_tensor), "r"(kCubeC), "r"(dim_w), "r"(kDimBlks),
        "r"(kCubeC), "r"(cube_w), "r"(kDimBlks), "l"(stride_w_bytes),
        "l"(static_cast<int64_t>(kCubeC) * 2), "r"(0), "r"(start_w), "r"(0),
        "r"(0)
      : "memory");
#else
  // ppu0010 build: inert stand-in, refused at runtime by the launcher.
  for (int i = threadIdx.x; i < kDimBlks * kTokGrp * kCubeC; i += 64)
    dst_shared[i] = __float2bfloat16(0.0f);
#endif
}

__device__ __forceinline__ void step4_dsa_paged_decode_copy_paged_group(
    __ppu_bfloat16* dst, const __ppu_bfloat16* src, int physical_tokens,
    int token, DsaPagedKvLayout layout) {
  const bool valid =
      static_cast<unsigned>(token) < static_cast<unsigned>(physical_tokens);
  const int page = valid ? token >> layout.page_shift : 0;
  const int offset = valid ? token & (layout.page_size - 1) : layout.page_size;
  step4_dsa_paged_decode_copy_group(
      dst, src + static_cast<int64_t>(page) * layout.page_stride,
      layout.page_size, offset, 8, layout.token_stride * 2);
}

template <int SPLITS>
__global__
__launch_bounds__(64, 2) void step4_dsa_paged_decode_shared_pairs_kernel(
    const __ppu_bfloat16* query, const __ppu_bfloat16* key,
    const __ppu_bfloat16* value, const int32_t* regions, const int32_t* counts,
    float* maxima, float* denominators, float* accumulators, int kv_heads,
    int region_k, int physical_tokens, float scale,
    DsaPagedKvLayout kv_layout) {
  using namespace awmma;
  constexpr int R = 4;
  constexpr int NT = 2;
  using Stage = __ppu_bfloat16[R][3][8][64];
  __shared__ __align__(128) Stage qstage[2];
  const int warp = __all_sync(0xffffffffu, threadIdx.x >= 32) ? 1 : 0;
  const int lane = threadIdx.x & 31;
  const int head = lane >> 2;
  const int t0 = lane & 3;
  const int lr = lane & 7;
  const int lu = lane >> 3;
  const int task = (blockIdx.x % 32) * 8 + blockIdx.x / 32;
  const int pair = task / SPLITS;
  const int qt = (pair / 2) * 4 + (pair & 1) + warp * 2;
  const int kh = 0;
  const int original_n = region_k;
  const int cap = region_k * 2;
  const int common_n = counts[pair * 2 + 1];
  const int total_n = counts[pair * 2];
  region_k = common_n;
  const int split = task % SPLITS;
  const int tiles = (region_k + R - 1) / R;
  const int begin = (tiles * split / SPLITS) * R;
  const int count = min(region_k, (tiles * (split + 1) / SPLITS) * R) - begin;
  const int64_t rb = static_cast<int64_t>(pair) * 3 * cap;
  fragment<matrix_a, 16, 16, 16, __ppu_bfloat16, row_major> qf[12];
  const auto* qp =
      query + (static_cast<int64_t>(qt) * kv_heads * 16 + kh * 16) * 192;
#pragma unroll
  for (int t = 0; t < 2; ++t)
    step4_dsa_paged_decode_copy_group(&qstage[warp][t][0][0][0], qp, 16, t * 8,
                                      8, 384);
  const int first_slot = begin + lane;
  const int first_id =
      first_slot < region_k && lane < count ? regions[rb + first_slot] : -1;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int id = __shfl_sync(0xffffffffu, first_id, r);
    if (warp == 0 && id >= 0 && id < physical_tokens / 8)
      step4_dsa_paged_decode_copy_paged_group(
          &qstage[r >> 1][2 + (r & 1)][0][0][0],
          key + kh * kv_layout.head_stride, physical_tokens, id * 8, kv_layout);
  }
  asm volatile("ppu.cp.async.commit_group;\n\t" ::: "memory");
  asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
#pragma unroll
  for (int k = 0; k < 12; ++k) {
    const int qc = (((k % 4) * 16 + 8 * (lu >> 1)) >> 3) ^ lr;
    ldm_x4(qf[k], &qstage[warp][lu & 1][k / 4][lr][qc * 8]);
  }
  fragment<accumulator, 16, 16, 16, float> out[12];
#pragma unroll
  for (int j = 0; j < 12; ++j) fill_fragment(out[j], 0.0f);
  float maximum[2] = {-INFINITY, -INFINITY};
  float denom[2] = {0, 0};
  int cr = -1, cs = -1;
  __syncthreads();
  for (int b = 0; b < count; b += R) {
    if ((b & 31) == 0) {
      const int slot = begin + b + lane;
      const bool valid = slot < region_k && b + lane < count;
      cr = valid ? regions[rb + slot] : -1;
      cs = valid ? regions[rb + (warp + 1) * cap + slot] : -1;
    }
    int ids[R], logical[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      ids[r] = __shfl_sync(0xffffffffu, cr, (b + r) & 31);
      logical[r] = __shfl_sync(0xffffffffu, cs, (b + r) & 31);
      if (b + r >= count || begin + b + r >= region_k) ids[r] = -1;
    }
    if (b > 0) {
      if (warp == 0)
        asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
      __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int vid = __shfl_sync(0xffffffffu, cr, (b + r) & 31);
      if (warp == 0)
        step4_dsa_paged_decode_copy_paged_group(
            &qstage[r >> 1][r & 1][0][0][0], value + kh * kv_layout.head_stride,
            physical_tokens,
            b + r < count && vid >= 0 && vid < physical_tokens / 8
                ? vid * 8
                : physical_tokens,
            kv_layout);
    }
    if (warp == 0) asm volatile("ppu.cp.async.commit_group;\n\t" ::: "memory");
    bool active_lane = false, all_active_lane = true;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const bool valid =
          logical[r] > 0 && ids[r] >= 0 && ids[r] < physical_tokens / 8;
      active_lane |= valid;
      all_active_lane &= valid;
    }
    const bool active = __any_sync(0xffffffffu, active_lane);
    const bool all_active = __all_sync(0xffffffffu, all_active_lane);
    uint32_t token_valid = 0, nan_mask = 0;
#pragma unroll
    for (int r = 0; r < R; ++r)
      if (logical[r] <= 0) nan_mask |= 1u << r;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int token = t * 16 + frag_col(t0, i);
        const int r = token / 8;
        if (ids[r] >= 0 && ids[r] < physical_tokens / 8 &&
            token % 8 < logical[r])
          token_valid |= 1u << (t * 8 + i);
      }
    }
    fragment<matrix_a, 16, 16, 16, __ppu_bfloat16, row_major> pf[NT];
    float rescale[2];
    if (active) {
      fragment<accumulator, 16, 16, 16, float> scores[NT];
      fragment<matrix_b, 16, 16, 16, __ppu_bfloat16, col_major> kf;
#pragma unroll
      for (int t = 0; t < NT; ++t) fill_fragment(scores[t], 0.0f);
#pragma unroll
      for (int k = 0; k < 12; ++k) {
        const int chunk = (((k % 4) * 16 + 8 * (lu & 1)) >> 3) ^ lr;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          ldm_x4(kf, &qstage[t][2 + (lu >> 1)][k / 4][lr][chunk * 8]);
          mma_sync(scores[t], qf[k], kf, scores[t]);
        }
      }
      float lm[2] = {-INFINITY, -INFINITY};
      uint32_t valid_bits = 0;
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const bool valid = token_valid & (1u << (t * 8 + i));
          if (valid) valid_bits |= 1u << (t * 8 + i);
          const float s = valid ? scores[t].x[i] : -INFINITY;
          scores[t].x[i] = s;
          lm[frag_token(i)] = fmaxf(lm[frag_token(i)], s);
        }
      }
      float next[2];
#pragma unroll
      for (int tau = 0; tau < 2; ++tau) {
        const float tm = head_max(lm[tau]);
        const bool valid = tm > -INFINITY;
        next[tau] = valid ? fmaxf(maximum[tau], tm) : maximum[tau];
        rescale[tau] =
            valid ? __expf((maximum[tau] - next[tau]) * scale) : 1.0f;
      }

      float sum[2] = {0, 0};
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
          const int tau = frag_token(i);
          const float offset = next[tau] == -INFINITY ? 0.0f : next[tau];
          float x0 = (scores[t].x[i] - offset) * scale;
          float x1 = (scores[t].x[i + 1] - offset) * scale;
          if (!(scale > 0)) {
            if (!(valid_bits & (1u << (t * 8 + i)))) x0 = -INFINITY;
            if (!(valid_bits & (1u << (t * 8 + i + 1)))) x1 = -INFINITY;
          }
          const float p0 = __expf(x0), p1 = __expf(x1);
          const __ppu_bfloat162 packed = __floats2bfloat162_rn(p0, p1);
          reinterpret_cast<uint32_t*>(&pf[t].x[0])[i / 2] =
              *reinterpret_cast<const uint32_t*>(&packed);
          sum[tau] += p0;
          sum[tau] += p1;
        }
      }
#pragma unroll
      for (int tau = 0; tau < 2; ++tau) {
        denom[tau] = denom[tau] * rescale[tau] + sum[tau];
        maximum[tau] = next[tau];
      }
    }
    if (warp == 0) asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
    __syncthreads();
    if (b + R < count) {
      const int slot = begin + b + R + lane;
      const int next_cr =
          ((b + R) & 31) == 0
              ? (slot < region_k && b + R + lane < count ? regions[rb + slot]
                                                         : -1)
              : cr;
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const int id = __shfl_sync(0xffffffffu, next_cr, (b + R + r) & 31);
        if (warp == 0 && b + R + r < count && id >= 0 &&
            id < physical_tokens / 8)
          step4_dsa_paged_decode_copy_paged_group(
              &qstage[r >> 1][2 + (r & 1)][0][0][0],
              key + kh * kv_layout.head_stride, physical_tokens, id * 8,
              kv_layout);
      }
      if (warp == 0)
        asm volatile("ppu.cp.async.commit_group;\n\t" ::: "memory");
    }
    if (active) {
      fragment<matrix_b, 16, 16, 16, __ppu_bfloat16, row_major> vf;
#pragma unroll
      for (int j = 0; j < 12; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
          out[j].x[i] *= frag_token(i) == 0 ? rescale[0] : rescale[1];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          const int chunk = (((j % 4) * 16 + 8 * (lu >> 1)) >> 3) ^ lr;
          ldm_x4_trans(vf, &qstage[t][lu & 1][j / 4][lr][chunk * 8]);
          if (!all_active) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int r = t * 2 + (i & 1);
              if (nan_mask & (1u << r))
                reinterpret_cast<uint32_t*>(&vf.x[0])[i] = 0;
            }
          }
          mma_sync(out[j], pf[t], vf, out[j]);
        }
      }
    }
    if (b + R >= count) __syncthreads();
  }
  {
    Stage& sk = qstage[warp];
    Stage& sv = sk;
    const int private_begin = warp == 0 ? common_n : original_n;
    const int private_end = warp == 0 ? original_n : total_n;
    const int n = private_end - private_begin;
    const int tiles = (n + R - 1) / R;
    const int begin = private_begin + (tiles * split / SPLITS) * R;
    const int count =
        min(private_end, private_begin + (tiles * (split + 1) / SPLITS) * R) -
        begin;
    const int region_k = private_end;
    int cr = -1, cs = -1;
    for (int b = 0; b < count; b += R) {
      if ((b & 31) == 0) {
        const int slot = begin + b + lane;
        const bool valid = slot < region_k && b + lane < count;
        cr = valid ? regions[rb + slot] : -1;
        cs = valid ? regions[rb + (warp + 1) * cap + slot] : -1;
      }
      int ids[R], logical[R];
#pragma unroll
      for (int r = 0; r < R; ++r) {
        ids[r] = __shfl_sync(0xffffffffu, cr, (b + r) & 31);
        logical[r] = __shfl_sync(0xffffffffu, cs, (b + r) & 31);
        if (b + r >= count || begin + b + r >= region_k) ids[r] = -1;
        if (ids[r] >= 0 && ids[r] < physical_tokens / 8 && logical[r] > 0)
          step4_dsa_paged_decode_copy_paged_group(
              &sk[r][0][0][0], key + kh * kv_layout.head_stride,
              physical_tokens, ids[r] * 8, kv_layout);
      }
      asm volatile("ppu.cp.async.commit_group;\n\t" ::: "memory");
      asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
      fragment<accumulator, 16, 16, 16, float> scores[NT];
      fragment<matrix_b, 16, 16, 16, __ppu_bfloat16, col_major> kf;
#pragma unroll
      for (int t = 0; t < NT; ++t) fill_fragment(scores[t], 0.0f);
#pragma unroll
      for (int k = 0; k < 12; ++k) {
        const int chunk = (((k % 4) * 16 + 8 * (lu & 1)) >> 3) ^ lr;
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          ldm_x4(kf, &sk[t * 2 + (lu >> 1)][k / 4][lr][chunk * 8]);
          mma_sync(scores[t], qf[k], kf, scores[t]);
        }
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const bool valid =
            ids[r] >= 0 && ids[r] < physical_tokens / 8 && logical[r] > 0;
        step4_dsa_paged_decode_copy_paged_group(
            &sv[r][0][0][0], value + kh * kv_layout.head_stride,
            physical_tokens, valid ? ids[r] * 8 : physical_tokens, kv_layout);
      }
      asm volatile("ppu.cp.async.commit_group;\n\t" ::: "memory");
      float lm[2] = {-INFINITY, -INFINITY};
      uint32_t valid_bits = 0;
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int token = t * 16 + frag_col(t0, i);
          const int r = token / 8;
          const bool valid = ids[r] >= 0 && ids[r] < physical_tokens / 8 &&
                             token % 8 < logical[r];
          if (valid) valid_bits |= 1u << (t * 8 + i);
          const float s = valid ? scores[t].x[i] : -INFINITY;
          scores[t].x[i] = s;
          lm[frag_token(i)] = fmaxf(lm[frag_token(i)], s);
        }
      }
      float next[2], rescale[2];
#pragma unroll
      for (int tau = 0; tau < 2; ++tau) {
        const float tm = head_max(lm[tau]);
        const bool valid = tm > -INFINITY;
        next[tau] = valid ? fmaxf(maximum[tau], tm) : maximum[tau];
        rescale[tau] =
            valid ? __expf((maximum[tau] - next[tau]) * scale) : 1.0f;
      }
      fragment<matrix_a, 16, 16, 16, __ppu_bfloat16, row_major> pf[NT];
      float sum[2] = {0, 0};
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
          const int tau = frag_token(i);
          const float offset = next[tau] == -INFINITY ? 0.0f : next[tau];
          float x0 = (scores[t].x[i] - offset) * scale;
          float x1 = (scores[t].x[i + 1] - offset) * scale;
          if (!(scale > 0)) {
            if (!(valid_bits & (1u << (t * 8 + i)))) x0 = -INFINITY;
            if (!(valid_bits & (1u << (t * 8 + i + 1)))) x1 = -INFINITY;
          }
          const float p0 = __expf(x0), p1 = __expf(x1);
          const __ppu_bfloat162 packed = __floats2bfloat162_rn(p0, p1);
          reinterpret_cast<uint32_t*>(&pf[t].x[0])[i / 2] =
              *reinterpret_cast<const uint32_t*>(&packed);
          sum[tau] += p0;
          sum[tau] += p1;
        }
      }
#pragma unroll
      for (int tau = 0; tau < 2; ++tau) {
        denom[tau] = denom[tau] * rescale[tau] + sum[tau];
        maximum[tau] = next[tau];
      }
      asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
#pragma unroll
      for (int j = 0; j < 12; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
          out[j].x[i] *= frag_token(i) == 0 ? rescale[0] : rescale[1];
      }
      fragment<matrix_b, 16, 16, 16, __ppu_bfloat16, row_major> vf;
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int j = 0; j < 12; ++j) {
          const int chunk = (((j % 4) * 16 + 8 * (lu >> 1)) >> 3) ^ lr;
          ldm_x4_trans(vf, &sv[t * 2 + (lu & 1)][j / 4][lr][chunk * 8]);
          mma_sync(out[j], pf[t], vf, out[j]);
        }
      }
    }
  }
#pragma unroll
  for (int tau = 0; tau < 2; ++tau) {
    const float d = head_sum(denom[tau]);
    const int64_t scalar =
        ((static_cast<int64_t>(qt) * kv_heads + kh) * 16 + head + tau * 8) *
            SPLITS +
        split;
    if (t0 == 0) {
      maxima[scalar] = d > 0 ? maximum[tau] * scale : -INFINITY;
      denominators[scalar] = d;
    }
#pragma unroll
    for (int j = 0; j < 12; ++j) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int i = tau * 2 + half * 4;
        const int64_t dst =
            (((static_cast<int64_t>(qt) * SPLITS + split) * 12 + j) * 4 +
             tau * 2 + half) *
                32 +
            lane;
        __stcg(reinterpret_cast<float2*>(accumulators) + dst,
               make_float2(out[j].x[i], out[j].x[i + 1]));
      }
    }
  }
}

template <int HEAD_TILES, int SPLITS, bool FUSED_HEADS = false,
          int PAGED_TILE_S = 32>
__global__ __launch_bounds__(64, 2) void step4_dsa_paged_decode_kernel(
    const __ppu_bfloat16* query, const __ppu_bfloat16* key,
    const __ppu_bfloat16* value, const int32_t* physical_regions,
    const int32_t* logical_starts, const int64_t* query_positions,
    __ppu_bfloat16* output, float* split_max, float* split_denom,
    float* split_acc, int q_len, int kv_heads, int region_k,
    int physical_tokens, float scale, DsaPagedKvLayout kv_layout) {
  constexpr int bytes = 2 * PAGED_TILE_S * kHeadDim * 2;
  __shared__ __align__(128) unsigned char scratch[bytes];
  attention_body<HEAD_TILES, SPLITS, FUSED_HEADS, PAGED_TILE_S>(
      query, key, value, physical_regions, query_positions, output,
      kv_heads * kQHeads * HEAD_TILES * kHeadDim, kHeadDim, kv_heads * region_k,
      region_k, kv_heads * kQHeads * HEAD_TILES * kHeadDim, kHeadDim, kv_heads,
      region_k, physical_tokens, q_len, scale, kv_layout, scratch, blockIdx.x,
      logical_starts, split_max, split_denom, split_acc);
}

template <int SPLITS>
__global__ void step4_dsa_paged_decode_merge(const float* split_max,
                                             const float* split_denom,
                                             const float* split_acc,
                                             __ppu_bfloat16* output,
                                             int total_heads) {
  const int head = blockIdx.x;
  if (head >= total_heads || threadIdx.x >= kHeadDim) return;
  float max_logit = -INFINITY;
  float denom = 0.0f;
  float acc = 0.0f;
#pragma unroll
  for (int s = 0; s < SPLITS; ++s) {
    const int scalar = head * SPLITS + s;
    const float local_denom = split_denom[scalar];
    if (local_denom > 0.0f) {
      const float local_max = split_max[scalar];
      const float new_max = fmaxf(max_logit, local_max);
      const float old_scale =
          max_logit > -INFINITY ? __expf(max_logit - new_max) : 0.0f;
      const float local_scale = __expf(local_max - new_max);
      denom = denom * old_scale + local_denom * local_scale;
      acc = acc * old_scale +
            split_acc[static_cast<int64_t>(scalar) * kHeadDim + threadIdx.x] *
                local_scale;
      max_logit = new_max;
    }
  }
  output[static_cast<int64_t>(head) * kHeadDim + threadIdx.x] =
      __float2bfloat16(denom > 0.0f ? acc / denom : 0.0f);
}

template <int SPLITS>
__global__ void step4_dsa_paged_decode_merge_vector(const float* split_max,
                                                    const float* split_denom,
                                                    const float* split_acc,
                                                    __ppu_bfloat16* output,
                                                    int total_heads) {
  const int head = blockIdx.x;
  if (head >= total_heads || threadIdx.x >= kHeadDim / 4) return;
  float max_logit = -INFINITY, denom = 0.0f;
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
  for (int s = 0; s < SPLITS; ++s) {
    const int scalar = head * SPLITS + s;
    const float local_denom = split_denom[scalar];
    if (local_denom > 0.0f) {
      const float local_max = split_max[scalar];
      const float new_max = fmaxf(max_logit, local_max);
      const float old_scale =
          max_logit > -INFINITY ? __expf(max_logit - new_max) : 0.0f;
      const float local_scale = __expf(local_max - new_max);
      const float4 x = reinterpret_cast<const float4*>(
          split_acc + static_cast<int64_t>(scalar) * kHeadDim)[threadIdx.x];
      denom = denom * old_scale + local_denom * local_scale;
      acc.x = acc.x * old_scale + x.x * local_scale;
      acc.y = acc.y * old_scale + x.y * local_scale;
      acc.z = acc.z * old_scale + x.z * local_scale;
      acc.w = acc.w * old_scale + x.w * local_scale;
      max_logit = new_max;
    }
  }
  const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
  auto* dst = reinterpret_cast<__ppu_bfloat162*>(
      output + static_cast<int64_t>(head) * kHeadDim + threadIdx.x * 4);
  dst[0] = __floats2bfloat162_rn(acc.x * inv, acc.y * inv);
  dst[1] = __floats2bfloat162_rn(acc.z * inv, acc.w * inv);
}

struct DecodeLaunchConfig {
  int splits;
  bool fuse_heads;
  bool compact;
  bool warp_specialized;
};

DecodeLaunchConfig decode_launch_config(int64_t q_len, int64_t q_heads,
                                        int64_t kv_heads, int64_t region_k) {
  const int64_t programs =
      q_len * kv_heads * (q_heads == kv_heads * 16 ? 2 : 1);
  const bool fuse_heads = q_heads == kv_heads * 16 &&
                          (q_len >= 8 || (q_len == 4 && region_k >= 256)) &&
                          region_k > 32;
  int splits;
  if (region_k <= 32) {
    splits = 4;
  } else if (programs >= 128) {
    splits = 1;
  } else if (programs >= 64) {
    splits = 2;
  } else if (programs >= 32) {
    splits = 4;
  } else if (programs >= 16) {
    splits = 8;
  } else {
    splits = 16;
  }
  if (fuse_heads && q_len >= 8) splits *= 2;
  if (fuse_heads && q_len * kv_heads >= 64 && q_len * kv_heads < 128)
    splits = 4;
  const bool compact =
      fuse_heads && q_len == 64 && kv_heads == 1 && region_k >= 256;
  const bool warp_specialized =
      fuse_heads && q_len == 4 && kv_heads == 1 && region_k >= 512;
  if (compact) splits = 8;
  if (warp_specialized) splits = 32;
  return {splits, fuse_heads, compact, warp_specialized};
}

template <int SPLITS>
void launch_decode_merge(const tvm::ffi::TensorView split_max,
                         const tvm::ffi::TensorView split_denom,
                         const tvm::ffi::TensorView split_acc,
                         const tvm::ffi::TensorView output,
                         hggcStream_t stream) {
  const int heads = output.size(0) * output.size(1);
  auto* out = static_cast<__ppu_bfloat16*>(output.data_ptr());
  if (heads >= 512) {
    step4_dsa_paged_decode_merge_vector<SPLITS><<<heads, 64, 0, stream>>>(
        static_cast<const float*>(split_max.data_ptr()),
        static_cast<const float*>(split_denom.data_ptr()),
        static_cast<const float*>(split_acc.data_ptr()), out, heads);
  } else {
    step4_dsa_paged_decode_merge<SPLITS><<<heads, 256, 0, stream>>>(
        static_cast<const float*>(split_max.data_ptr()),
        static_cast<const float*>(split_denom.data_ptr()),
        static_cast<const float*>(split_acc.data_ptr()), out, heads);
  }
}

}  // namespace

namespace sglang {
namespace step5_dsa {

// TensorMatcher / SymbolicSize / SymbolicDevice / LaunchKernel live in host::.
using namespace host;
namespace {

// Fold the per-split (max, denominator) states into one natural LSE. The max is
// already scaled by the softmax scale, so no extra factor enters here.
__global__ void reduce_split_lse_kernel(const float* __restrict__ split_max,
                                        const float* __restrict__ split_denom,
                                        float* __restrict__ lse, int64_t pairs,
                                        int splits) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pairs) return;
  const int64_t base = index * splits;
  float maximum = -INFINITY;
  for (int s = 0; s < splits; ++s)
    maximum = fmaxf(maximum, split_max[base + s]);
  if (maximum == -INFINITY) {
    lse[index] = -INFINITY;
    return;
  }
  float denominator = 0.0f;
  for (int s = 0; s < splits; ++s) {
    if (split_denom[base + s] > 0.0f)
      denominator +=
          split_denom[base + s] * __expf(split_max[base + s] - maximum);
  }
  lse[index] = denominator > 0.0f ? maximum + __logf(denominator) : -INFINITY;
}

}  // namespace

void paged_decode(const tvm::ffi::TensorView query,
                  const tvm::ffi::TensorView key_cache,
                  const tvm::ffi::TensorView value_cache,
                  const tvm::ffi::TensorView physical_regions,
                  const tvm::ffi::TensorView logical_starts,
                  const tvm::ffi::TensorView query_positions,
                  const tvm::ffi::TensorView output,
                  const tvm::ffi::TensorView lse,
                  const tvm::ffi::TensorView split_max,
                  const tvm::ffi::TensorView split_denom,
                  const tvm::ffi::TensorView split_acc, double scaling) {
  // The tc02 kernels exist only on the ppu0015 core, and the ppu0010 build of
  // this file is inert, so refuse that core instead of returning zeros.
  CHECK_HOST(host::runtime::get_cc(query.device().device_id) >= 89)
      << "Step5 sparse paged decode requires the ppu0015 PPU core";
  check_scale(scaling);
  auto S_query = SymbolicSize{"query"};
  auto S_qheads = SymbolicSize{"q_heads"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({S_query, S_qheads, 192})
      .with_dtype<__ppu_bfloat16>()
      .with_device<kDLCUDA>(device)
      .ensure_alignment(4)
      .verify(query);
  // SGLang hands K and V over as independent native planes; the paged layout
  // below is identical for both, so one check covers the pair.
  auto S_pages = SymbolicSize{"pages"};
  auto S_page = SymbolicSize{"page_size"};
  auto S_kvheads = SymbolicSize{"kv_heads"};
  TensorMatcher({S_pages, S_page, S_kvheads, 192})
      .with_dtype<__ppu_bfloat16>()
      .with_device<kDLCUDA>(device)
      .with_strides({-1, -1, -1, 1})
      .ensure_alignment(16)
      .verify(key_cache)
      .verify(value_cache);
  CHECK_HOST(key_cache.stride(0) > 0 && key_cache.stride(1) > 0 &&
             key_cache.stride(2) > 0)
      << "cache page, token and head strides must be positive";
  auto S_regions = SymbolicSize{"regions"};
  TensorMatcher({S_query, S_kvheads, S_regions})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device)
      .verify(physical_regions)
      .verify(logical_starts);
  TensorMatcher({S_query})
      .with_dtype<int64_t>()
      .with_device<kDLCUDA>(device)
      .verify(query_positions);
  TensorMatcher({S_query, S_qheads, 192})
      .with_dtype<__ppu_bfloat16>()
      .with_device<kDLCUDA>(device)
      .ensure_alignment(4)
      .verify(output);
  TensorMatcher({S_query, S_qheads})
      .with_dtype<float>()
      .with_device<kDLCUDA>(device)
      .verify(lse);
  check_int32({S_query.unwrap(), S_qheads.unwrap(), S_kvheads.unwrap(),
               S_regions.unwrap(), S_page.unwrap(),
               S_pages.unwrap() * S_page.unwrap(), query.numel(),
               physical_regions.numel()});
  CHECK_HOST(S_pages.unwrap() > 0) << "cache must contain at least one page";
  const int q_len = static_cast<int>(S_query.unwrap());
  const int q_heads = static_cast<int>(S_qheads.unwrap());
  const int kv_heads = static_cast<int>(S_kvheads.unwrap());
  const int region_k = static_cast<int>(S_regions.unwrap());
  const int page_size = static_cast<int>(S_page.unwrap());
  CHECK_HOST(q_len > 0 && region_k > 0 && kv_heads > 0 &&
             (q_heads == kv_heads * 8 || q_heads == kv_heads * 16))
      << "expected nonempty GQA8 or GQA16 decode batch";
  CHECK_HOST(page_size >= 8 && page_size % 8 == 0 &&
             ((page_size / 8) & (page_size / 8 - 1)) == 0)
      << "page size must contain a power-of-two number of 8-token regions";
  const auto config = decode_launch_config(q_len, q_heads, kv_heads, region_k);
  const int splits = config.splits;
  const bool fuse_heads = config.fuse_heads;
  const bool compact = config.compact;
  const bool warp_specialized = config.warp_specialized;
  const bool shared_pairs = compact && region_k == 513;
  TensorMatcher({S_query, S_qheads, splits})
      .with_dtype<float>()
      .with_device<kDLCUDA>(device)
      .verify(split_max)
      .verify(split_denom);
  TensorMatcher({S_query, S_qheads, splits, 192})
      .with_dtype<float>()
      .with_device<kDLCUDA>(device)
      .ensure_alignment(16)
      .verify(split_acc);
  check_int32(
      {int64_t{q_len} * q_heads * splits, int64_t{q_len} * kv_heads * splits});
  hggcStream_t stream = LaunchKernel::resolve_device(query.device());
  const dim3 grid(q_len * kv_heads * splits, 1,
                  q_heads == kv_heads * 16 && !fuse_heads ? 2 : 1);
  const int physical_tokens = key_cache.size(0) * page_size;
  const auto* key = static_cast<const __ppu_bfloat16*>(key_cache.data_ptr());
  const auto* value =
      static_cast<const __ppu_bfloat16*>(value_cache.data_ptr());
  int page_shift = 0;
  for (int n = page_size; n > 1; n >>= 1) ++page_shift;
  const DsaPagedKvLayout kv_layout = {
      page_size, page_shift, key_cache.stride(0), key_cache.stride(1),
      key_cache.stride(2)};
  const auto* q = static_cast<const __ppu_bfloat16*>(query.data_ptr());
  auto* out = static_cast<__ppu_bfloat16*>(output.data_ptr());
  const auto* regions_ptr =
      static_cast<const int32_t*>(physical_regions.data_ptr());
  const auto* starts_ptr =
      static_cast<const int32_t*>(logical_starts.data_ptr());
  const auto* positions_ptr =
      static_cast<const int64_t*>(query_positions.data_ptr());
  // Split scratch is written by the decode and merge kernels, so these stay
  // writable pointers even though the tensor handles are passed by value.
  auto* split_max_ptr = static_cast<float*>(split_max.data_ptr());
  auto* split_denom_ptr = static_cast<float*>(split_denom.data_ptr());
  auto* split_acc_ptr = static_cast<float*>(split_acc.data_ptr());
#define LAUNCH_DECODE(H, S, F, T)                                              \
  step4_dsa_paged_decode_kernel<H, S, F, T><<<grid, 64, 0, stream>>>(          \
      q, key, value, regions_ptr, starts_ptr, positions_ptr, out,              \
      split_max_ptr, split_denom_ptr, split_acc_ptr, q_len, kv_heads,          \
      region_k, physical_tokens, static_cast<float>(scaling), kv_layout)
  if (q_heads == kv_heads * 8) {
    if (splits == 1)
      LAUNCH_DECODE(1, 1, false, 48);
    else if (splits == 2)
      LAUNCH_DECODE(1, 2, false, 48);
    else if (splits == 4)
      LAUNCH_DECODE(1, 4, false, 48);
    else if (splits == 8)
      LAUNCH_DECODE(1, 8, false, 48);
    else
      LAUNCH_DECODE(1, 16, false, 48);
  } else if (shared_pairs) {
    const auto i32_dtype = DLDataType{kDLInt, 32, 1};
    auto pair_metadata = host::ffi::empty({q_len / 2, 3, region_k * 2},
                                          i32_dtype, device.unwrap());
    auto pair_counts =
        host::ffi::empty({q_len / 2, 2}, i32_dtype, device.unwrap());
    if (physical_tokens / 8 <= (1 << 21)) {
      step4_dsa_paged_decode_pair_metadata<true>
          <<<q_len / 2, kDsaPairMetadataThreads, 0, stream>>>(
              regions_ptr, starts_ptr, positions_ptr,
              static_cast<int32_t*>(pair_metadata.data_ptr()),
              static_cast<int32_t*>(pair_counts.data_ptr()),
              physical_tokens / 8);
    } else {
      step4_dsa_paged_decode_pair_metadata<false>
          <<<q_len / 2, kDsaPairMetadataThreads, 0, stream>>>(
              regions_ptr, starts_ptr, positions_ptr,
              static_cast<int32_t*>(pair_metadata.data_ptr()),
              static_cast<int32_t*>(pair_counts.data_ptr()),
              physical_tokens / 8);
    }
    step4_dsa_paged_decode_shared_pairs_kernel<8>
        <<<q_len / 2 * splits, 64, 0, stream>>>(
            q, key, value, static_cast<int32_t*>(pair_metadata.data_ptr()),
            static_cast<int32_t*>(pair_counts.data_ptr()), split_max_ptr,
            split_denom_ptr, split_acc_ptr, kv_heads, region_k, physical_tokens,
            static_cast<float>(scaling), kv_layout);
  } else if (compact) {
    step4_dsa_paged_decode_compact_kernel<8><<<grid, 32, 0, stream>>>(
        q, key, value, regions_ptr, starts_ptr, positions_ptr, split_max_ptr,
        split_denom_ptr, split_acc_ptr, kv_heads, region_k, physical_tokens,
        static_cast<float>(scaling), kv_layout);
  } else if (warp_specialized) {
    step4_dsa_paged_decode_ws_kernel<32, 32><<<grid, 64, 0, stream>>>(
        q, key, value, regions_ptr, starts_ptr, positions_ptr, split_max_ptr,
        split_denom_ptr, split_acc_ptr, kv_heads, region_k, physical_tokens,
        static_cast<float>(scaling), kv_layout);
  } else if (fuse_heads) {
    if (splits == 2)
      LAUNCH_DECODE(2, 2, true, 32);
    else if (splits == 4)
      LAUNCH_DECODE(2, 4, true, 32);
    else if (splits == 8)
      LAUNCH_DECODE(2, 8, true, 32);
    else
      LAUNCH_DECODE(2, 16, true, 32);
  } else {
    if (region_k <= 32) {
      LAUNCH_DECODE(2, 4, false, 48);
    } else {
      if (splits == 4)
        LAUNCH_DECODE(2, 4, false, 32);
      else if (splits == 8)
        LAUNCH_DECODE(2, 8, false, 32);
      else
        LAUNCH_DECODE(2, 16, false, 32);
    }
  }
#undef LAUNCH_DECODE
  if (shared_pairs) {
    step4_dsa_paged_decode_merge_native<8><<<q_len * 12, 64, 0, stream>>>(
        split_max_ptr, split_denom_ptr, split_acc_ptr, out);
  } else if (warp_specialized) {
    step4_dsa_paged_decode_merge_parallel<32, 8>
        <<<q_len * q_heads, 256, 0, stream>>>(split_max_ptr, split_denom_ptr,
                                              split_acc_ptr, out);
  } else {
    switch (splits) {
      case 1:
        break;  // The attention kernel writes output directly.
      case 2:
        launch_decode_merge<2>(split_max, split_denom, split_acc, output,
                               stream);
        break;
      case 4:
        launch_decode_merge<4>(split_max, split_denom, split_acc, output,
                               stream);
        break;
      case 8:
        launch_decode_merge<8>(split_max, split_denom, split_acc, output,
                               stream);
        break;
      case 16:
        launch_decode_merge<16>(split_max, split_denom, split_acc, output,
                                stream);
        break;
    }
  }
  const int64_t lse_pairs = static_cast<int64_t>(q_len) * q_heads;
  constexpr int kLseThreads = 128;
  reduce_split_lse_kernel<<<(lse_pairs + kLseThreads - 1) / kLseThreads,
                            kLseThreads, 0, stream>>>(
      split_max_ptr, split_denom_ptr, static_cast<float*>(lse.data_ptr()),
      lse_pairs, splits);
  CHECK_CUDA(hggcGetLastError()) << "Step5 DSA paged decode launch failed";
}

int64_t paged_decode_num_splits(int64_t q_len, int64_t q_heads,
                                int64_t kv_heads, int64_t region_k) {
  return decode_launch_config(q_len, q_heads, kv_heads, region_k).splits;
}

}  // namespace step5_dsa
}  // namespace sglang
