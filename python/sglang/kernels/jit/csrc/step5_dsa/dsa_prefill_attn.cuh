// SPDX-License-Identifier: Apache-2.0
//
// Step5 sparse prefill attention over SGLang-owned native KV storage.
//
// Ported from the Step5 prefill kernel. The original stages two or four queries
// per tile and sorts each row's selected regions in shared memory, which both
// rest on the same assumption: a region's physical KV row and its logical
// position are the same number. SGLang pages its KV pool, so a region owns eight
// slots at an arbitrary pool offset while its causal limit comes from the
// selection's logical start. This port keeps the single-query staging tile and
// reads both planes: physical regions address K/V, logical starts drive the
// causal mask. The multi-query tiles, the per-row slot sort, the dead-tile skip
// and the bitmap reorders are dropped together with the dense-logical ids that
// sized their bitmaps.

#pragma once

#include <cmath>
#include <cstdint>

#include <hggc_bf16.h>
#include <mma.h>

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
constexpr int kCubeC = 64;
constexpr int kDimBlks = kHeadDim / kCubeC;
constexpr int kTilesPerBlk = kCubeC / kMma;
constexpr int kTokGrp = 8;
constexpr int kRegionBlock = 8;
// Staged K/V tokens per tile; four regions of eight tokens each.
constexpr int kBlockS = 32;
constexpr int kTokTiles = kBlockS / kMma;
constexpr int kTokGrps = kBlockS / kTokGrp;
constexpr int kRegionsPerTile = kBlockS / kRegionBlock;
constexpr int kDimTilesPerWarp = kKSteps / 2;
constexpr int kThreads = 2 * kWarpSize;
constexpr int kStageElems = kTokGrps * kDimBlks * kTokGrp * kCubeC;
constexpr int kStageBytes = kStageElems * sizeof(__ppu_bfloat16);
constexpr int kScratchBytes = 2 * kStageBytes;
static_assert(kRegionBlock == kTokGrp, "one token group stages one region");
static_assert(kRegionsPerTile == kTokGrps, "one staging tile holds whole regions");

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

// Stage one region: eight KV rows of `kCubeC` dims per block. PADZ zero-fills
// rows outside `dim_w`, so a region that points past the pool reads zeros.
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
                 "r"(0), "r"(start_w), "r"(0),
                 "r"(0)  // swzlMode 0 = 128 B
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
  for (int i = threadIdx.x; i < kDimBlks * kTokGrp * kCubeC; i += kThreads)
    dst_shared[i] = __float2bfloat16(0.0f);
}
#endif

// Reduce across the four lanes that share a head (t0 == head).
__device__ __forceinline__ float head_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
}

__device__ __forceinline__ float head_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v + __shfl_xor_sync(0xffffffffu, v, 2);
}

// Column of accumulator/matrix_a element `i` for this lane, i.e. `n` for the QK
// output and `k` for the PV A operand.  Row is `t1 + 8*v1`.
__device__ __forceinline__ int frag_col(int t0, int i) {
  return 2 * t0 + (i & 1) + 8 * ((i >> 2) & 1);
}

__device__ __forceinline__ bool frag_is_real_row(int i) {
  return ((i >> 1) & 1) == 0;
}

__device__ __forceinline__ int vmask_bit(int ti, int i) {
  return ti * 4 + (i & 1) + 2 * ((i >> 2) & 1);
}

// One CTA owns one query token of one KV head and its whole selection. HEAD_TILES
// is 1 for eight query heads per KV head and 2 for sixteen, where blockIdx.z
// picks which eight of them this CTA computes.
template <int HEAD_TILES, typename PosT>
__device__ __forceinline__ void attention_body(
    const __ppu_bfloat16* __restrict__ query,
    const __ppu_bfloat16* __restrict__ key,
    const __ppu_bfloat16* __restrict__ value,
    const int32_t* __restrict__ region_ids,
    const int32_t* __restrict__ region_starts,
    const int32_t* __restrict__ live_regions,
    const PosT* __restrict__ query_positions, __ppu_bfloat16* __restrict__ output,
    float* __restrict__ lse, int64_t stride_qt, int64_t stride_qh,
    int64_t stride_kt, int64_t stride_kh, int64_t stride_vt, int64_t stride_vh,
    int64_t stride_rq, int64_t stride_rh, int64_t stride_ot, int64_t stride_oh,
    int q_heads, int kv_heads, int region_k, int kv_len, float scale,
    unsigned char* scratch, int program) {
  using namespace awmma;
  using Stage = __ppu_bfloat16[kTokGrps][kDimBlks][kTokGrp][kCubeC];
  auto& sK = *reinterpret_cast<Stage*>(scratch);
  auto& sV = *reinterpret_cast<Stage*>(scratch + kStageBytes);

  const int q_token = program / kv_heads;
  const int kv_head = program - q_token * kv_heads;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int t0 = lane & 3;
  // This lane's head for every fragment row v1 == 0.
  const int head = lane >> 2;
  const int lr = lane & 7;
  const int lu = lane >> 3;
  const int a_col = 8 * (lu >> 1);
  const int b_col = 8 * (lu & 1);
  const int swzl_row = lr;

  const int64_t qpos = query_positions[q_token];
  const int64_t rid_base = static_cast<int64_t>(q_token) * stride_rq +
                           static_cast<int64_t>(kv_head) * stride_rh;
  // Slots past this row's selection are -1 anyway; bounding the walk keeps a
  // short prompt from paying for the full 513-slot window.
  const int owned = live_regions[q_token];
  const int walk = owned < region_k ? owned : region_k;
  const int region_limit = kv_len / kRegionBlock;

  fragment<accumulator, kMma, kMma, kMma, float> oacc[kDimTilesPerWarp];
#pragma unroll
  for (int i = 0; i < kDimTilesPerWarp; ++i) fill_fragment(oacc[i], 0.0f);
  // The two warps split the 192 output dims; both compute the same scores.
  const int dim0 = warp * kDimTilesPerWarp * kMma;

  float running_max = -INFINITY;
  float denom = 0.0f;

  fragment<matrix_a, kMma, kMma, kMma, __ppu_bfloat16, row_major> qfrag[kKSteps];
  {
    const __ppu_bfloat16* qrow =
        query + static_cast<int64_t>(q_token) * stride_qt +
        (static_cast<int64_t>(kv_head) * (kQHeads * HEAD_TILES) +
         (HEAD_TILES == 1 ? 0 : static_cast<int>(blockIdx.z) * kQHeads) +
         head) *
            stride_qh;
#pragma unroll
    for (int k = 0; k < kKSteps; ++k) {
      fill_fragment(qfrag[k], __float2bfloat16(0.0f));
      *reinterpret_cast<__ppu_bfloat162*>(&qfrag[k].x[0]) =
          *reinterpret_cast<const __ppu_bfloat162*>(qrow + k * kMma + 2 * t0);
      *reinterpret_cast<__ppu_bfloat162*>(&qfrag[k].x[4]) =
          *reinterpret_cast<const __ppu_bfloat162*>(qrow + k * kMma + 2 * t0 + 8);
    }
  }

  for (int slot0 = 0; slot0 < walk; slot0 += kRegionsPerTile) {
    int rids[kRegionsPerTile];
    int64_t starts[kRegionsPerTile];
#pragma unroll
    for (int rr = 0; rr < kRegionsPerTile; ++rr) {
      const int slot = slot0 + rr;
      if (slot >= walk) {
        rids[rr] = -1;
        starts[rr] = -1;
      } else {
        const int rid = region_ids[rid_base + static_cast<int64_t>(slot)];
        // A region outside the pool means the slot is stale, not merely empty;
        // dropping it keeps the zero-padded copy out of the softmax.
        rids[rr] = static_cast<unsigned>(rid) < static_cast<unsigned>(region_limit)
                       ? rid
                       : -1;
        starts[rr] = region_starts[rid_base + static_cast<int64_t>(slot)];
      }
    }

    // Warp 0 stages K and warp 1 stages V, one region per token group.
    {
      const int copy_warp = warp;
      const __ppu_bfloat16* tensor =
          (copy_warp == 0 ? key : value) +
          static_cast<int64_t>(kv_head) *
              (copy_warp == 0 ? stride_kh : stride_vh);
      const int64_t stride_bytes = (copy_warp == 0 ? stride_kt : stride_vt) * 2;
      __ppu_bfloat16* dstbuf =
          (copy_warp == 0 ? &sK[0][0][0][0] : &sV[0][0][0][0]);
      constexpr int kGrpElems = kDimBlks * kTokGrp * kCubeC;
#pragma unroll
      for (int tg = 0; tg < kTokGrps; ++tg) {
        const int r = rids[tg];
        // A negative region means "no slot". Clamping to region 0 keeps the
        // copy in bounds; the visibility mask discards the result.
        const int start = r >= 0 ? r * kRegionBlock : 0;
        aiu_copy_group(dstbuf + tg * kGrpElems, tensor, kv_len, start, kTokGrp,
                       stride_bytes);
      }
      asm volatile("ppu.cp.async.commit_group;\n\t" ::: "memory");
    }

    // Token t lives in region t / kRegionBlock; its causal position is the
    // region's logical start plus the offset inside it.
    unsigned vmask = 0;
#pragma unroll
    for (int ti = 0; ti < kTokTiles; ++ti) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        if (!frag_is_real_row(i)) continue;
        const int t = ti * kMma + frag_col(t0, i);
        const int rr = t / kRegionBlock;
        const int64_t tok = starts[rr] + (t % kRegionBlock);
        if (rids[rr] >= 0 && starts[rr] >= 0 && tok <= qpos)
          vmask |= 1u << vmask_bit(ti, i);
      }
    }

    asm volatile("ppu.cp.async.wait_group 0;\n\t" ::: "memory");
    __syncthreads();

    // QK^T over the current staging tile; col_major reads K transposed.
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
          // The unit's eight token rows are ti*16 + 8*(lu>>1) .. +7, which is
          // exactly token group ti*2 + (lu>>1), row lr inside it.
          ldm_x4(kfrag, &sK[ti * 2 + (lu >> 1)][kdb][lr][kchunk << 3]);
          mma_sync(sacc[ti], qfrag[k], kfrag, sacc[ti]);
        }
      }
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
    const float new_max = has_valid ? fmaxf(running_max, tile_max) : running_max;
    const float rescale =
        has_valid ? __expf((running_max - new_max) * scale) : 1.0f;

    // Probabilities go straight into the PV A operand: the accumulator and
    // matrix_a element maps are identical, so element i here is element i
    // there.
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
        // Storing as bf16 *is* the rounding the reference applies before its PV
        // dot, and the denominator accumulates the same rounded values.
        const __ppu_bfloat16 pb = __float2bfloat16(p);
        pfrag[ti].x[i] = pb;
        local_sum += __bfloat162float(pb);
      }
    }
    denom = denom * rescale + local_sum;
    running_max = new_max;

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
          // dim0 is warp * 96, which is not a multiple of kCubeC, so the block
          // and inner offset are computed from the absolute dim.
          const int vdim = dim0 + j * kMma;
          const int vdb = vdim / kCubeC;
          const int vchunk = (((vdim % kCubeC) + a_col) >> 3) ^ swzl_row;
          // A-style address form, so the token group is ti*2 + (lu&1).
          ldm_x4_trans(vfrag, &sV[ti * 2 + (lu & 1)][vdb][lr][vchunk << 3]);
          mma_sync(oacc[j], pfrag[ti], vfrag, oacc[j]);
        }
      }
    }
    __syncthreads();
  }

  const float d = head_sum(denom);
  const float inv = d > 0.0f ? 1.0f / d : 0.0f;
  const int64_t gh = static_cast<int64_t>(kv_head) * (kQHeads * HEAD_TILES) +
                     (HEAD_TILES == 1 ? 0
                                      : static_cast<int>(blockIdx.z) * kQHeads) +
                     head;
  const int64_t o_base = static_cast<int64_t>(q_token) * stride_ot +
                         gh * stride_oh;
#pragma unroll
  for (int j = 0; j < kDimTilesPerWarp; ++j) {
    const int64_t base = o_base + dim0 + j * kMma;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      // i = 0,1 for half 0 (columns 2*t0, 2*t0+1); i = 4,5 for half 1.
      const int i0 = half * 4;
      const __ppu_bfloat162 pair =
          __floats2bfloat162_rn(oacc[j].x[i0] * inv, oacc[j].x[i0 + 1] * inv);
      *reinterpret_cast<__ppu_bfloat162*>(output + base + 2 * t0 + 8 * half) =
          pair;
    }
  }
  // Both warps hold the same statistics; one lane per head publishes them. The
  // running maximum is unscaled, so the natural LSE is max * scale + log d.
  if (warp == 0 && t0 == 0) {
    lse[static_cast<int64_t>(q_token) * q_heads + gh] =
        (running_max == -INFINITY || d <= 0.0f)
            ? -INFINITY
            : running_max * scale + __logf(d);
  }
}

template <int HEAD_TILES, typename PosT>
__global__ __launch_bounds__(kThreads, 8) void step4_dsa_prefill_attn_kernel(
    const __ppu_bfloat16* query, const __ppu_bfloat16* key,
    const __ppu_bfloat16* value, const int32_t* region_ids,
    const int32_t* region_starts, const int32_t* live_regions,
    const PosT* query_positions, __ppu_bfloat16* output, float* lse,
    int64_t stride_qt, int64_t stride_qh, int64_t stride_kt, int64_t stride_kh,
    int64_t stride_vt, int64_t stride_vh, int64_t stride_rq, int64_t stride_rh,
    int64_t stride_ot, int64_t stride_oh, int q_heads, int kv_heads,
    int region_k, int kv_len, float scale) {
  __shared__ __align__(128) unsigned char scratch[kScratchBytes];
  attention_body<HEAD_TILES, PosT>(
      query, key, value, region_ids, region_starts, live_regions,
      query_positions, output, lse, stride_qt, stride_qh, stride_kt, stride_kh,
      stride_vt, stride_vh, stride_rq, stride_rh, stride_ot, stride_oh, q_heads,
      kv_heads, region_k, kv_len, scale, scratch, static_cast<int>(blockIdx.x));
}

}  // namespace

namespace sglang {
namespace step5_dsa {

// TensorMatcher / SymbolicSize / SymbolicDevice / LaunchKernel live in host::.
using namespace host;
namespace {

// Rows with an empty live prefix still need zeros and a -inf LSE; the AOT
// host used output.zero_() / lse.fill_() for that, which have no TensorView
// equivalent, so a tiny reset kernel covers both planes in one launch.
__global__ void reset_prefill_outputs_kernel(
    __ppu_bfloat16* __restrict__ output, float* __restrict__ lse,
    int64_t out_elems, int64_t lse_elems) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < out_elems) output[index] = __float2bfloat16(0.0f);
  if (index < lse_elems) lse[index] = -INFINITY;
}

}  // namespace

void prefill_attn(const tvm::ffi::TensorView query,
                  const tvm::ffi::TensorView key,
                  const tvm::ffi::TensorView value,
                  const tvm::ffi::TensorView physical_regions,
                  const tvm::ffi::TensorView logical_starts,
                  const tvm::ffi::TensorView live_regions,
                  const tvm::ffi::TensorView query_positions,
                  const tvm::ffi::TensorView output,
                  const tvm::ffi::TensorView lse, double scaling) {
  // The tc02 kernels exist only on the ppu0015 core, and the ppu0010 build of
  // this file is inert, so refuse that core instead of returning zeros.
  CHECK_HOST(host::runtime::get_cc(query.device().device_id) >= 89)
      << "Step5 sparse prefill attention requires the ppu0015 PPU core";
  check_scale(scaling);
  auto S_query = SymbolicSize{"query"};
  auto S_qheads = SymbolicSize{"q_heads"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  // The Q fragments load two adjacent bf16 straight from global, so the row
  // start of every (token, head) pair must land on a 4 B boundary; the
  // alignment check below covers both the pointer and the row strides.
  TensorMatcher({S_query, S_qheads, kHeadDim})
      .with_dtype<__ppu_bfloat16>()
      .with_device<kDLCUDA>(device)
      .with_strides({-1, -1, 1})
      .ensure_alignment(4)
      .verify(query);
  // The 128-bit K/V loads need row strides that are multiples of 8 bf16 and
  // 16 B row starts; one alignment check covers both planes and the pointer.
  auto S_kvlen = SymbolicSize{"kv_len"};
  auto S_kvheads = SymbolicSize{"kv_heads"};
  TensorMatcher({S_kvlen, S_kvheads, kHeadDim})
      .with_dtype<__ppu_bfloat16>()
      .with_device<kDLCUDA>(device)
      .with_strides({-1, -1, 1})
      .ensure_alignment(16)
      .verify(key)
      .verify(value);
  TensorMatcher({S_query, S_qheads, kHeadDim})
      .with_dtype<__ppu_bfloat16>()
      .with_device<kDLCUDA>(device)
      .with_strides({-1, -1, 1})
      .ensure_alignment(4)
      .verify(output);
  auto S_regions = SymbolicSize{"regions"};
  TensorMatcher({S_query, S_kvheads, S_regions})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device)
      .verify(physical_regions)
      .verify(logical_starts);
  TensorMatcher({S_query})
      .with_dtype<int32_t>()
      .with_device<kDLCUDA>(device)
      .verify(live_regions);
  TensorMatcher({S_query})
      .with_dtype<int32_t, int64_t>()
      .with_device<kDLCUDA>(device)
      .verify(query_positions);
  TensorMatcher({S_query, S_qheads})
      .with_dtype<float>()
      .with_device<kDLCUDA>(device)
      .verify(lse);

  const int64_t q_len = S_query.unwrap();
  const int64_t q_heads = S_qheads.unwrap();
  const int64_t kv_heads = S_kvheads.unwrap();
  const int64_t region_k = S_regions.unwrap();
  const int64_t kv_len = S_kvlen.unwrap();
  CHECK_HOST(kv_heads > 0 && (q_heads == kv_heads * kQHeads ||
                              q_heads == kv_heads * kQHeads * 2))
      << "specialised for 8 or 16 query heads per KV head, got q_heads="
      << q_heads << " kv_heads=" << kv_heads;
  CHECK_HOST(kv_len % kRegionBlock == 0)
      << "KV storage must hold whole 8-token regions";
  CHECK_HOST(region_k <= 1024) << "at most 1024 region slots are supported";
  check_int32({q_len, q_heads, kv_heads, kv_len, region_k, q_len * q_heads,
               q_len * kv_heads * region_k});
  if (q_len == 0) return;
  hggcStream_t stream = LaunchKernel::resolve_device(query.device());
  if (region_k == 0) {
    constexpr int kResetThreads = 256;
    const int64_t out_elems = query.numel();
    const int64_t lse_elems = lse.numel();
    const int64_t reset_work = out_elems > lse_elems ? out_elems : lse_elems;
    reset_prefill_outputs_kernel<<<static_cast<unsigned>(
                                       (reset_work + kResetThreads - 1) /
                                       kResetThreads),
                                   kResetThreads, 0, stream>>>(
        static_cast<__ppu_bfloat16*>(output.data_ptr()),
        static_cast<float*>(lse.data_ptr()), out_elems, lse_elems);
    CHECK_CUDA(hggcGetLastError()) << "Step5 DSA prefill reset launch failed";
    return;
  }

  const auto* q = static_cast<const __ppu_bfloat16*>(query.data_ptr());
  const auto* k = static_cast<const __ppu_bfloat16*>(key.data_ptr());
  const auto* v = static_cast<const __ppu_bfloat16*>(value.data_ptr());
  auto* out = static_cast<__ppu_bfloat16*>(output.data_ptr());
  // lse is an output of the softmax epilogue, so it must stay writable.
  auto* lse_ptr = static_cast<float*>(lse.data_ptr());
  const auto* regions_ptr =
      static_cast<const int32_t*>(physical_regions.data_ptr());
  const auto* starts_ptr =
      static_cast<const int32_t*>(logical_starts.data_ptr());
  const auto* live_ptr = static_cast<const int32_t*>(live_regions.data_ptr());
  const auto* pos_long_ptr =
      static_cast<const int64_t*>(query_positions.data_ptr());
  const auto* pos_int_ptr =
      static_cast<const int32_t*>(query_positions.data_ptr());
  const dim3 grid(static_cast<unsigned>(q_len * kv_heads), 1,
                  q_heads == kv_heads * kQHeads * 2 ? 2 : 1);
  const bool pos_long = is_type<int64_t>(query_positions.dtype());

#define LAUNCH_PREFILL(HEAD_TILES)                                            \
  do {                                                                        \
    if (pos_long)                                                             \
      step4_dsa_prefill_attn_kernel<HEAD_TILES, int64_t>                      \
          <<<grid, kThreads, 0, stream>>>(                                    \
              q, k, v, regions_ptr, starts_ptr, live_ptr, pos_long_ptr, out,  \
              lse_ptr, query.stride(0), query.stride(1), key.stride(0),       \
              key.stride(1), value.stride(0), value.stride(1),                \
              physical_regions.stride(0), physical_regions.stride(1),         \
              output.stride(0), output.stride(1),                             \
              static_cast<int>(q_heads), static_cast<int>(kv_heads),          \
              static_cast<int>(region_k), static_cast<int>(kv_len),           \
              static_cast<float>(scaling));                                   \
    else                                                                      \
      step4_dsa_prefill_attn_kernel<HEAD_TILES, int32_t>                      \
          <<<grid, kThreads, 0, stream>>>(                                    \
              q, k, v, regions_ptr, starts_ptr, live_ptr, pos_int_ptr, out,   \
              lse_ptr, query.stride(0), query.stride(1), key.stride(0),       \
              key.stride(1), value.stride(0), value.stride(1),                \
              physical_regions.stride(0), physical_regions.stride(1),         \
              output.stride(0), output.stride(1),                             \
              static_cast<int>(q_heads), static_cast<int>(kv_heads),          \
              static_cast<int>(region_k), static_cast<int>(kv_len),           \
              static_cast<float>(scaling));                                   \
  } while (0)

  if (grid.z == 2) {
    LAUNCH_PREFILL(2);
  } else {
    LAUNCH_PREFILL(1);
  }
#undef LAUNCH_PREFILL
  CHECK_CUDA(hggcGetLastError())
      << "Step5 DSA prefill attention launch failed";
}

}  // namespace step5_dsa
}  // namespace sglang