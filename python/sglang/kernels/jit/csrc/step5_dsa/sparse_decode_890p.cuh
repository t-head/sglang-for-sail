// SPDX-License-Identifier: Apache-2.0
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <algorithm>
#include <cmath>
#include <cfloat>

#include "step5_dsa_device.cuh"

namespace sglang::step5_decode {
using namespace step5_dsa;

constexpr int kMaxSplits = 32;
constexpr int kMergeWarps = 4;
constexpr int kSingleWarpRows = 16;
constexpr int kDirectRows = 4096;

struct Decode {
  const __ppu_bfloat16 *q, *k, *v;
  const int64_t *packed;
  const int32_t *lengths, *counts, *valid_rows;
  __ppu_bfloat16 *out;
  float *lse, *partial, *partial_lse;
  int rows, heads, slots, packed_stride, physical_regions, splits, capacity;
  float scale;
};

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
  for (int d = 16; d; d >>= 1)
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, d));
  return x;
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int d = 16; d; d >>= 1)
    x += __shfl_xor_sync(0xffffffffu, x, d);
  return x;
}

template <int Heads, int TG, int Warps, bool Direct>
__global__ __launch_bounds__(32 * Warps, Warps == 1 ? 4 : 1) void packed_attention(Decode p) {
  constexpr int NT = TG / 2;
  constexpr int Halves = Heads / 8;
  constexpr float kLog2e = 1.4426950408889634f;
  constexpr float kLn2 = 0.6931471805599453f;
  const int warp = Warps == 1 ? 0 : __builtin_ppu_to_uniform_b32(threadIdx.x / 32);
  const int block = blockIdx.x * Warps + warp;
  if constexpr (Warps > 1) {
    if (block >= (Direct ? p.rows : p.rows * p.splits))
      return;
  }
  const int row = Direct ? block : Warps == 1 ? block / p.splits : block % p.rows;
  const int split = Direct ? 0 : Warps == 1 ? block % p.splits : block / p.rows;
  const int lane = threadIdx.x & 31;
  const int t0 = lane & 3, lr = lane & 7, lu = lane >> 3, head = lane >> 2;
  const int a_col = 8 * (lu >> 1), b_col = 8 * (lu & 1);
  const int tiles = (p.slots + TG - 1) / TG;
  const int stride = (p.slots + p.splits - 1) / p.splits;
  const int begin = Direct ? 0 : Warps == 1 ? split * stride : (split * tiles / p.splits) * TG;
  const int end = Direct ? p.slots : Warps == 1 ? min(p.slots, begin + stride)
      : min(p.slots, ((split + 1) * tiles / p.splits) * TG);
  const int64_t pos = int64_t(p.lengths[row]) - 1;
  const int live = row < p.valid_rows[0] && pos >= 0 ? p.counts[row] : 0;
  const int width = min(end, live) - begin;
  const auto *packed = p.packed + int64_t(row) * p.packed_stride;

  __shared__ __align__(128) KvStage<TG> key_stage[Warps], value_stage[Warps];
  auto &sk = key_stage[warp];
  auto &sv = value_stage[warp];
  MmaOperand qfrag[kHeadMmaTiles];
  MmaAccumulator output[kHeadMmaTiles];
#pragma unroll
  for (int d = 0; d < kHeadMmaTiles; ++d) {
    clear_fragment(qfrag[d]);
    clear_fragment(output[d]);
#pragma unroll
    for (int tau = 0; tau < Halves; ++tau) {
      const auto *src = p.q + (int64_t(row) * Heads + head + tau * 8) * kHeadDim;
      qfrag[d].x[tau] = *reinterpret_cast<const uint32_t *>(src + 16 * d + 2 * t0);
      qfrag[d].x[tau + 2] =
          *reinterpret_cast<const uint32_t *>(src + 16 * d + 2 * t0 + 8);
    }
  }

  float maximum[Halves], denom[Halves];
#pragma unroll
  for (int tau = 0; tau < Halves; ++tau) {
    maximum[tau] = Warps == 1 ? -INFINITY : -FLT_MAX;
    denom[tau] = 0.f;
  }
  const float scale2 = p.scale * kLog2e;
  int region = 0, visible = 0;
  for (int base = 0; base < width; base += TG) {
    if ((base & 31) == 0) {
      const int64_t raw = base + lane < width ? packed[begin + base + lane] : -1;
      const int id = int(raw >> 32);
      const int64_t start = uint32_t(raw);
      const bool valid = id >= 0 && id < p.physical_regions && start <= pos;
      visible = valid ? (pos - start >= 7 ? 8 : int(pos - start) + 1) : 0;
      region = valid ? id : 0;
      if constexpr (Direct) {
        if (!__any_sync(0xffffffffu, visible != 0)) {
          base += 32 - TG;
          continue;
        }
      }
    }
    int ids[TG], counts[TG];
    bool nonempty = false, full = true;
#pragma unroll
    for (int r = 0; r < TG; ++r) {
      ids[r] = __builtin_ppu_to_uniform_b32(
          __shfl_sync(0xffffffffu, region, (base & 31) + r));
      counts[r] = __builtin_ppu_to_uniform_b32(
          __shfl_sync(0xffffffffu, visible, (base & 31) + r));
      nonempty |= counts[r] != 0;
      full &= counts[r] == kRegionTokens;
    }
    if (!nonempty)
      continue;
#pragma unroll
    for (int r = 0; r < TG; ++r) {
      const int start = counts[r] == 0 ? kRegionTokens : 0;
      const int64_t offset = int64_t(ids[r]) * kRegionTokens * kHeadDim;
      aiu_copy_group(&sk[r][0][0][0], p.k + offset, counts[r] + start,
                     start, kRegionTokens);
      aiu_copy_group(&sv[r][0][0][0], p.v + offset, counts[r] + start,
                     start, kRegionTokens);
    }
    asm volatile("ppu.cp.async.commit_group;" ::: "memory");
    asm volatile("ppu.cp.async.wait_group 0;" ::: "memory");
    __syncwarp();

    MmaAccumulator scores[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t)
      clear_fragment(scores[t]);
#pragma unroll
    for (int d = 0; d < kHeadMmaTiles; ++d) {
      const int chunk = (((d % 4) * 16 + b_col) >> 3) ^ lr;
      MmaOperand key[NT];
#pragma unroll
      for (int t = 0; t < NT; ++t)
        ldm_x4(key[t], &sk[2 * t + (lu >> 1)][d / 4][lr][chunk * 8]);
#pragma unroll
      for (int t = 0; t < NT; ++t)
        mma_bf16(scores[t], qfrag[d], key[t], scores[t]);
    }
    float peak[Halves];
#pragma unroll
    for (int tau = 0; tau < Halves; ++tau)
      peak[tau] = -INFINITY;
    if (Warps > 1 && full) {
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int tau = (i >> 1) & 1;
          if (tau < Halves)
            peak[tau] = fmaxf(peak[tau], scores[t].x[i]);
        }
      }
    } else {
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int tau = (i >> 1) & 1;
          const bool valid = tau < Halves && 2 * t0 + (i & 1) < counts[2 * t + (i >> 2)];
          scores[t].x[i] = valid ? scores[t].x[i] : -INFINITY;
          if (tau < Halves)
            peak[tau] = fmaxf(peak[tau], scores[t].x[i]);
        }
      }
    }
    if constexpr (Warps == 1) {
      float rescale[Halves];
#pragma unroll
      for (int tau = 0; tau < Halves; ++tau) {
        const float tile_peak = head_max(peak[tau]) * scale2;
        const float next = fmaxf(maximum[tau], tile_peak);
        rescale[tau] = next > -INFINITY ? __exp2f(maximum[tau] - next) : 1.f;
        denom[tau] *= rescale[tau];
        maximum[tau] = next;
      }
#pragma unroll
      for (int d = 0; d < kHeadMmaTiles; ++d) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int tau = (i >> 1) & 1;
          if (tau < Halves)
            output[d].x[i] *= rescale[tau];
        }
      }
    } else {
      bool increase = false;
#pragma unroll
      for (int tau = 0; tau < Halves; ++tau)
        increase |= peak[tau] * scale2 > maximum[tau];
      if (__any_sync(0xffffffffu, increase)) {
        float rescale[Halves];
#pragma unroll
        for (int tau = 0; tau < Halves; ++tau) {
          const float local = head_max(peak[tau]) * scale2;
          const float bound = maximum[tau] == -FLT_MAX ? local : ceilf(local * 0.125f) * 8.f;
          const float next = local > maximum[tau] ? bound : maximum[tau];
          rescale[tau] = __exp2f(maximum[tau] - next);
          denom[tau] *= rescale[tau];
          maximum[tau] = next;
        }
#pragma unroll
        for (int d = 0; d < kHeadMmaTiles; ++d) {
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const int tau = (i >> 1) & 1;
            if (tau < Halves)
              output[d].x[i] *= rescale[tau];
          }
        }
      }
    }
    MmaOperand prob[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const int tau = (i >> 1) & 1;
        float a = 0.f, b = 0.f;
        if (tau < Halves && (Warps > 1 || maximum[tau] > -INFINITY)) {
          a = __exp2f(fmaf(scores[t].x[i], scale2, -maximum[tau]));
          b = __exp2f(fmaf(scores[t].x[i + 1], scale2, -maximum[tau]));
          denom[tau] += a + b;
        }
        *reinterpret_cast<__ppu_bfloat162 *>(&prob[t].x[i / 2]) =
            __floats2bfloat162_rn(a, b);
      }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int d = 0; d < kHeadMmaTiles; ++d) {
        MmaOperand value;
        const int chunk = (((d % 4) * 16 + a_col) >> 3) ^ lr;
        ldm_x4_trans(value, &sv[2 * t + (lu & 1)][d / 4][lr][chunk * 8]);
        mma_bf16(output[d], prob[t], value, output[d]);
      }
    }
    __syncwarp();
  }
#pragma unroll
  for (int tau = 0; tau < Halves; ++tau) {
    const int64_t h = int64_t(row) * Heads + head + tau * 8;
    const int64_t scalar = h * p.capacity + split;
    const float norm = head_sum(denom[tau]);
    const float inv = norm > 0 ? 1.f / norm : 0.f;
    const float lse = norm > 0 ? maximum[tau] * kLn2 + __logf(norm) : -INFINITY;
    if (t0 == 0) {
      if (Direct || p.splits == 1)
        p.lse[h] = lse;
      else
        p.partial_lse[scalar] = lse;
    }
    if constexpr (!Direct && Warps > 1) {
      // The warp merge skips empty splits and their unwritten partials.
      if (p.splits > 1 && norm == 0.f)
        continue;
    }
#pragma unroll
    for (int d = 0; d < kHeadMmaTiles; ++d) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int dim = d * 16 + 2 * t0 + half * 8;
        const int i = 2 * tau + 4 * half;
        const float a = output[d].x[i] * inv, b = output[d].x[i + 1] * inv;
        if (Direct || p.splits == 1)
          *reinterpret_cast<__ppu_bfloat162 *>(p.out + h * kHeadDim + dim) =
              __floats2bfloat162_rn(a, b);
        else
          *reinterpret_cast<float2 *>(p.partial + scalar * kHeadDim + dim) = make_float2(a, b);
      }
    }
  }
}

__global__ void merge_attention(Decode p) {
  const int64_t h = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t first = h * p.capacity;
  __shared__ float weights[kMaxSplits];
  if (tid < 32) {
    const float local = tid < p.splits ? p.partial_lse[first + tid] : -INFINITY;
    const float peak = warp_max(local);
    const float weight = peak > -INFINITY ? __expf(local - peak) : 0.f;
    const float sum = warp_sum(weight);
    weights[tid] = sum > 0 ? weight / sum : 0.f;
    if (tid == 0)
      p.lse[h] = sum > 0 ? peak + __logf(sum) : -INFINITY;
  }
  __syncthreads();
  if (tid >= kHeadDim / 4)
    return;
  float4 result = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int s = 0; s < p.splits; ++s) {
    const float4 data = reinterpret_cast<const float4 *>(p.partial + (first + s) * kHeadDim)[tid];
    const float w = weights[s];
    result.x = fmaf(w, data.x, result.x);
    result.y = fmaf(w, data.y, result.y);
    result.z = fmaf(w, data.z, result.z);
    result.w = fmaf(w, data.w, result.w);
  }
  auto *dst = reinterpret_cast<__ppu_bfloat162 *>(p.out + h * kHeadDim + tid * 4);
  dst[0] = __floats2bfloat162_rn(result.x, result.y);
  dst[1] = __floats2bfloat162_rn(result.z, result.w);
}

__global__ void merge_attention_warp(Decode p) {
  const int64_t h = int64_t(blockIdx.x) * kMergeWarps + threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int64_t first = h * p.capacity;
  const float local = lane < p.splits ? p.partial_lse[first + lane] : -INFINITY;
  const float peak = warp_max(local);
  const float weight = peak > -INFINITY ? __expf(local - peak) : 0.f;
  const float sum = warp_sum(weight);
  const float w = sum > 0 ? weight / sum : 0.f;
  if (lane == 0)
    p.lse[h] = sum > 0 ? peak + __logf(sum) : -INFINITY;
  uint32_t active = __ballot_sync(0xffffffffu, w != 0.f);
  float2 result[3] = {};
  while (active) {
    const int split = __ffs(active) - 1;
    const float scale = __shfl_sync(0xffffffffu, w, split);
    const auto *data = reinterpret_cast<const float2 *>(p.partial + (first + split) * kHeadDim) + lane;
#pragma unroll
    for (int d = 0; d < 3; ++d) {
      const float2 value = data[d * 32];
      result[d].x = fmaf(scale, value.x, result[d].x);
      result[d].y = fmaf(scale, value.y, result[d].y);
    }
    active &= active - 1;
  }
  auto *dst = reinterpret_cast<__ppu_bfloat162 *>(p.out + h * kHeadDim) + lane;
#pragma unroll
  for (int d = 0; d < 3; ++d)
    dst[d * 32] = __floats2bfloat162_rn(result[d].x, result[d].y);
}

template <int Warps, bool Direct = false>
void launch(Decode p, DLDevice device) {
  constexpr int TG = Warps == 1 ? 4 : 2;
  const int work = Direct ? p.rows : p.rows * p.splits;
  const int blocks = (work + Warps - 1) / Warps;
  host::LaunchKernel attention(blocks, 32 * Warps, device);
  if (p.heads == 16)
    attention(packed_attention<16, TG, Warps, Direct>, p);
  else
    attention(packed_attention<8, TG, Warps, Direct>, p);
  if (!Direct && p.splits > 1) {
    const int output_heads = p.rows * p.heads;
    if constexpr (Warps > 1)
      host::LaunchKernel(output_heads / kMergeWarps, 32 * kMergeWarps, device)(
          merge_attention_warp, p);
    else
      host::LaunchKernel(output_heads, 64, device)(merge_attention, p);
  }
}

int64_t split_capacity(int64_t rows) {
  if (rows >= kDirectRows) return 1;
  if (rows <= 4) return kMaxSplits;
  if (rows <= 32) return 16;
  if (rows <= 160) return 12;
  if (rows <= 320) return 8;
  return 4;
}

void run(
    tvm::ffi::TensorView q, tvm::ffi::TensorView k, tvm::ffi::TensorView v,
    tvm::ffi::TensorView packed, tvm::ffi::TensorView counts,
    tvm::ffi::TensorView lengths, tvm::ffi::TensorView valid_rows,
    tvm::ffi::TensorView partial, tvm::ffi::TensorView partial_lse,
    tvm::ffi::TensorView out, tvm::ffi::TensorView lse, double scale) {
  using namespace host;
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  auto rows = SymbolicSize{"rows"};
  auto heads = SymbolicSize{"heads"};
  auto slots = SymbolicSize{"KV slots"};
  auto width = SymbolicSize{"packed width"};
  auto capacity = SymbolicSize{"split capacity"};
  TensorMatcher({rows, heads, kHeadDim})
      .with_dtype<__ppu_bfloat16>().with_device(device).ensure_alignment(16).verify(q).verify(out);
  TensorMatcher({slots, 1, kHeadDim})
      .with_dtype<__ppu_bfloat16>().with_device(device).ensure_alignment(16).verify(k).verify(v);
  TensorMatcher({rows, width}).with_dtype<int64_t>().with_device(device).verify(packed);
  TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(counts).verify(lengths);
  TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(valid_rows);
  TensorMatcher({rows, heads}).with_dtype<float>().with_device(device).verify(lse);
  const auto n = rows.unwrap(), h = heads.unwrap(), s = slots.unwrap(), r = width.unwrap();
  RuntimeCheck(h == 8 || h == 16, "Step5 Decode expects 8 or 16 Q heads");
  RuntimeCheck(n <= INT32_MAX / kMaxSplits && s % 64 == 0 && s / kRegionTokens <= INT32_MAX);
  RuntimeCheck(r >= 1 && r <= 516 && std::isfinite(scale) && scale > 0);
  const int64_t scratch_rows = n >= kDirectRows ? 0 : n;
  TensorMatcher({scratch_rows, heads, capacity, kHeadDim})
      .with_dtype<float>().with_device(device).ensure_alignment(16).verify(partial);
  TensorMatcher({scratch_rows, heads, capacity}).with_dtype<float>().with_device(device).verify(partial_lse);
  const auto c = capacity.unwrap();
  RuntimeCheck(c >= 1 && c <= kMaxSplits);
  if (n == 0) return;

  const int selected = int(std::min<int64_t>(r, 513));
  const int partitions = n <= kSingleWarpRows ? selected : (selected + 1) / 2;
  const int splits = std::min({int(split_capacity(n)), int(c), partitions});
  const Decode p{
      static_cast<const __ppu_bfloat16 *>(q.data_ptr()),
      static_cast<const __ppu_bfloat16 *>(k.data_ptr()),
      static_cast<const __ppu_bfloat16 *>(v.data_ptr()),
      static_cast<const int64_t *>(packed.data_ptr()),
      static_cast<const int32_t *>(lengths.data_ptr()),
      static_cast<const int32_t *>(counts.data_ptr()),
      static_cast<const int32_t *>(valid_rows.data_ptr()),
      static_cast<__ppu_bfloat16 *>(out.data_ptr()),
      static_cast<float *>(lse.data_ptr()), static_cast<float *>(partial.data_ptr()),
      static_cast<float *>(partial_lse.data_ptr()), int(n), int(h), selected,
      int(r), int(s / kRegionTokens), splits, int(c), float(scale),
  };
  const auto target = device.unwrap();
  const tvm::ffi::CUDADeviceGuard guard(target.device_id);
  if (n >= kDirectRows)
    launch<2, true>(p, target);
  else if (n <= kSingleWarpRows)
    launch<1>(p, target);
  else
    launch<2>(p, target);
}

} // namespace sglang::step5_decode
