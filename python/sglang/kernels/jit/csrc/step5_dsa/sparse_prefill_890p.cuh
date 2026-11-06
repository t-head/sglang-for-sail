// SPDX-License-Identifier: Apache-2.0
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <cfloat>
#include <climits>
#include <cmath>

#include "step5_dsa_device.cuh"

namespace sglang::step5_prefill {

using namespace step5_dsa;

constexpr int kHeads = STEP5_PREFILL_HEADS;
static_assert(kHeads == 8 || kHeads == 16);

constexpr int kWidth = 513;
constexpr int kGroupRows = 16;
constexpr int kWindowRegions = 8192;
constexpr int kWords = kWindowRegions / 32;

struct Params {
  const __ppu_bfloat16 *q, *k, *v;
  __ppu_bfloat16 *out;
  float *lse;
  const int64_t *packed;
  const int32_t *live, *pos, *starts, *ends;
  int32_t *physical, *limits, *tile_counts;
  int rows, groups;
  int64_t capacity, qs, qh, os, oh;
  bool has_kv;
  float scale;

  __device__ int64_t tile_index(int group, int tile) const {
    return group * capacity + tile;
  }

  __device__ uint64_t selection(int row, int slot) const {
    const int64_t raw = packed[int64_t(row) * kWidth + slot];
    const int position = pos[row];
    const uint32_t start = uint32_t(raw);
    if (raw < 0 || position < 0 || start > uint32_t(position))
      return 0;
    const uint32_t n = min(8u, uint32_t(position) - start + 1u);
    return (uint64_t(raw) & 0xffffffff00000000ull) | n;
  }
  __device__ void write_slot(int row, int slot, uint32_t phys, uint32_t visible,
                             uint32_t max_visible) const {
    const int64_t idx = int64_t(row) * kWidth + slot;
    // The low four bits store the longest visible prefix in the tile.
    physical[idx] = (phys << 4) | max_visible;
    limits[idx] = visible;
  }
  __device__ uint64_t read_slot(int row, int slot) const {
    const int64_t idx = int64_t(row) * kWidth + slot;
    return (uint64_t(uint32_t(physical[idx])) << 32) | uint32_t(limits[idx]);
  }
  template <int BQ>
  __device__ void copy(__ppu_bfloat16 *dst, uint64_t entry, bool value) const {
    const uint32_t lookup = entry >> 32;
    const uint32_t phys = BQ == 1 ? lookup : lookup >> 4;
    const auto *src = (value ? v : k) + int64_t(phys) * 8 * 192;
    const int n = BQ == 1 ? int(uint32_t(entry)) : int(lookup & 15);
    // Put an empty cube beyond a nonzero extent so PADZ writes all zeros.
    const int start = n == 0 ? 8 : 0;
    aiu_copy_group(dst, src, n + start, start, kRegionTokens);
  }
};

template <int Words>
struct WindowState {
  int warp_prefix[Words / 32];
  int count;
  unsigned next_base;
};

template <int BQ, bool BalanceWords, int Words, int Threads = Words, bool RowMap = false>
__device__ __forceinline__ void collect_window(const Params &p, int first, int last,
                                              unsigned base, int written,
                                              WindowState<Threads> &state, int *tile_count) {
  extern __shared__ __align__(128) unsigned char temp[];
  auto *bitmap = reinterpret_cast<uint32_t (*)[Words]>(temp);
  auto *mapping = reinterpret_cast<int *>(bitmap + BQ);
  const int tid = threadIdx.x;
  const int query = first + tid % BQ;
  int last_pos = query < last ? p.pos[query] : -1;
#pragma unroll
  for (int stride = BQ / 2; stride; stride >>= 1)
    last_pos = max(last_pos, __shfl_xor_sync(0xffffffffu, last_pos, stride));
  const unsigned final_region = unsigned(max(0, last_pos)) / 8;
  const int used_words = final_region >= base
      ? min(Words, int((final_region - base) / 32) + 1) : 1;
  const int clear_log = 32 - __clz(unsigned(used_words - 1));
  const int clear_words = 1 << clear_log;
  for (int i = tid; i < BQ * clear_words; i += blockDim.x)
    bitmap[i >> clear_log][i & (clear_words - 1)] = 0;
  if (tid == 0)
    state.next_base = UINT_MAX;
  __syncthreads();
  unsigned next = UINT_MAX;
  const int warp = __builtin_ppu_to_uniform_b32(tid / 32);
  const int row = warp % BQ;
  const int owner = first + row;
  const int position = owner < last ? p.pos[owner] : -1;
  const int live = position < 0 ? 0 : min(kWidth, p.live[owner]);
  for (int slot = (warp / BQ) * 32 + (tid & 31); slot < live;
       slot += Threads / BQ) {
    const int64_t raw = p.packed[int64_t(owner) * kWidth + slot];
    const uint32_t start = raw;
    if (raw < 0 || start > uint32_t(position))
      continue;
    const unsigned region = start / 8;
    if (region < base)
      continue;
    const unsigned local = region - base;
    if (local >= (Words * 32)) {
      next = min(next, region & ~((Words * 32) - 1u));
      continue;
    }
    const int map_index = (RowMap ? row * Words * 32 : 0) + local;
    atomicExch(mapping + map_index, int(uint64_t(raw) >> 32));
    atomicOr(&bitmap[row][local / 32], 1u << (local % 32));
  }
  if (next != UINT_MAX)
    atomicMin(&state.next_base, next);
  __syncthreads();
  int word = tid;
  uint32_t slice = ~0u;
  if constexpr (BalanceWords) {
    // Divide a short causal prefix among the whole CTA, in logical order.
    const int log_bits = min(5, 32 - __clz(unsigned(max(0, last_pos)) / (8 * Threads)));
    const int bits = 1 << log_bits;
    word = tid >> (5 - log_bits);
    slice = (~0u >> (32 - bits)) << ((tid * bits) & 31);
  }
  uint32_t members[BQ], mask = 0;
#pragma unroll
  for (int r = 0; r < BQ; ++r) {
    members[r] = word < clear_words ? bitmap[r][word] & slice : 0;
    mask |= members[r];
  }
  const int count = __popc(mask);
  int inclusive = count;
#pragma unroll
  for (int step = 1; step < 32; step *= 2) {
    const int prev = __shfl_up_sync(0xffffffffu, inclusive, step);
    if ((tid & 31) >= step)
      inclusive += prev;
  }
  if ((tid & 31) == 31)
    state.warp_prefix[tid / 32] = inclusive;
  __syncthreads();
  if (tid == 0) {
    int sum = 0;
    for (int warp = 0; warp < int(blockDim.x) / 32; ++warp) {
      const int n = state.warp_prefix[warp];
      state.warp_prefix[warp] = sum;
      sum += n;
    }
    state.count = sum;
    if (state.next_base == UINT_MAX)
      *tile_count = written + sum;
  }
  __syncthreads();
  int offset = written + state.warp_prefix[tid / 32] + inclusive - count;
  uint32_t positions[BQ];
  uint32_t common = mask;
  int earliest = INT_MAX;
#pragma unroll
  for (int r = 0; r < BQ; ++r) {
    const bool active = first + r < last;
    const int pos = active ? p.pos[first + r] : (kHeads == 8 ? 0 : INT_MAX);
    positions[r] = uint32_t(pos);
    if constexpr (kHeads == 16 && BalanceWords) {
      if (active)
        common &= members[r];
      earliest = min(earliest, pos);
    }
  }
  const int active_rows = min(BQ, last - first);
  const uint32_t full = 0x88888888u >> (4 * (8 - active_rows));
  while (mask) {
    const int bit = __ffs(mask) - 1, region = word * 32 + bit;
    const uint32_t start = (base + region) * 8;
    uint32_t lengths = full, visible = 8;
    int map_index = region;
    if (kHeads != 16 || !BalanceWords || !(common & (1u << bit)) ||
        earliest < 0 || uint32_t(earliest) < start + 7) {
      lengths = visible = 0;
#pragma unroll
      for (int r = 0; r < BQ; ++r) {
        if (members[r] & (1u << bit)) {
          if constexpr (RowMap)
            map_index = r * Words * 32 + region;
          const uint32_t n = min(8u, positions[r] - start + 1);
          visible = max(visible, n);
          lengths |= n << (4 * r);
        }
      }
    }
    p.write_slot(first, offset++, mapping[map_index], lengths, visible);
    mask &= mask - 1;
  }
}

template <int BQ, bool BalanceWords, int Words>
__global__ void step5_prepare_union_kernel(Params p) {
  constexpr int Tiles = kGroupRows / BQ;
  const int g = BalanceWords ? blockIdx.x : blockIdx.x / Tiles;
  const int last = min(p.ends[g], p.rows);
  const int start_tile = BalanceWords ? 0 : blockIdx.x % Tiles;
  const int end_tile = BalanceWords ? Tiles : start_tile + 1;
  __shared__ WindowState<Words> state;
  for (int tile = start_tile; tile < end_tile; ++tile) {
    const int first = p.starts[g] + tile * BQ;
    if (first >= last)
      break;
    if (BalanceWords && tile)
      __syncthreads();
    int *tile_count = p.tile_counts + p.tile_index(g, tile);
    if constexpr (kHeads == 8 && BQ == 8 && !BalanceWords && Words == kWords) {
      constexpr int LocalRegions = 1024;
      if (p.capacity <= LocalRegions) {
        const int r = first + threadIdx.x;
        const bool fits = threadIdx.x >= BQ || r >= last || p.pos[r] < 8 * LocalRegions;
        if (__syncthreads_and(fits)) {
          collect_window<BQ, true, LocalRegions / 32, Words, true>(
              p, first, last, 0, 0, state, tile_count);
          continue;
        }
      }
    }
    collect_window<BQ, BalanceWords, Words>(p, first, last, 0, 0, state, tile_count);
    unsigned base = state.next_base;
    int written = state.count;
    while (base != UINT_MAX) {
      __syncthreads();
      collect_window<BQ, false, Words>(p, first, last, base, written, state, tile_count);
      base = state.next_base;
      written += state.count;
    }
  }
}

template <int BQ, int HT, int BS, int PARTS>
__global__
__launch_bounds__((BQ == 1 ? 1 : BQ / 2) * HT * 32 *
                  PARTS) void step5_paged_attention_kernel(Params p) {
  constexpr int TG = BS / 8, NT = BS / 16;
  static_assert(BQ == 1 || BQ == 4);
  static_assert((BQ == 1) == (PARTS > 1));
  extern __shared__ __align__(128) unsigned char scratch[];
  constexpr int warps_per_part = (BQ == 1 ? 1 : BQ / 2) * HT;
  const int part = threadIdx.x / (32 * warps_per_part);
  auto *stage = scratch + part * BS * 192 * 4;
  auto &sk = *reinterpret_cast<KvStage<TG> *>(stage);
  auto &sv = *reinterpret_cast<KvStage<TG> *>(stage + BS * 192 * 2);
  constexpr bool warp_local_stage = PARTS > 1 && warps_per_part == 1;
  const int g = blockIdx.x / (kGroupRows / BQ);
  const int tile = blockIdx.x % (kGroupRows / BQ);
  const int first = p.starts[g] + tile * BQ, end_row = min(p.rows, p.ends[g]);
  if (first >= end_row)
    return;
  const int warp = threadIdx.x / 32 % warps_per_part, lane = threadIdx.x % 32;
  constexpr int pairs = BQ == 1 ? 1 : BQ / 2;
  const int pair = warp % pairs, head_tile = warp / pairs;
  const int query_offset = pair * 2;
  const int row = first + query_offset;
  const int head = head_tile * 8 + (lane >> 2);
  const int t0 = lane & 3, lr = lane & 7, lu = lane >> 3;
  const int a_col = 8 * (lu >> 1), b_col = 8 * (lu & 1);

  const int total = BQ == 1
      ? (p.has_kv ? p.live[first] : 0)
      : p.tile_counts[p.tile_index(g, tile)];
  const int partition_stride = (total + PARTS - 1) / PARTS;
  const int offset = part * partition_stride;
  const int partition_size = BQ == 1
      ? max(0, min(partition_stride, total - offset)) : total;
  // A CTA barrier requires the same trip count across partitions.
  const int walk = warp_local_stage ? partition_size : partition_stride;
  MmaOperand qfrag[kHeadMmaTiles];
  MmaAccumulator out[kHeadMmaTiles];
#pragma unroll
  for (int k = 0; k < kHeadMmaTiles; ++k) {
    clear_fragment(qfrag[k]);
    clear_fragment(out[k]);
#pragma unroll
    for (int tau = 0; tau < 2; ++tau) {
      if (row + tau < end_row && (BQ > 1 || tau == 0)) {
        const auto *src = p.q + (row + tau) * p.qs + head * p.qh + k * 16 + 2 * t0;
        qfrag[k].x[tau] = *reinterpret_cast<const uint32_t *>(src);
        qfrag[k].x[tau + 2] = *reinterpret_cast<const uint32_t *>(src + 8);
      }
    }
  }
  float maximum[2] = {-INFINITY, -INFINITY}, denominator[2] = {0, 0};
  for (int base = 0; base < walk; base += TG) {
    uint64_t entries[TG];
#pragma unroll
    for (int r = 0; r < TG; ++r) {
      uint64_t entry = 0;
      if (base + r < partition_size) {
        if constexpr (BQ == 1)
          entry = p.selection(first, offset + base + r);
        else
          entry = p.read_slot(first, base + r);
      }
      entries[r] = entry;
    }
    if (warp == 0) {
#pragma unroll
      for (int r = 0; r < TG; ++r)
        p.copy<BQ>(&sk[r][0][0][0], entries[r], false);
      asm volatile("ppu.cp.async.commit_group;" ::: "memory");
#pragma unroll
      for (int r = 0; r < TG; ++r)
        p.copy<BQ>(&sv[r][0][0][0], entries[r], true);
      asm volatile("ppu.cp.async.commit_group;" ::: "memory");
      asm volatile("ppu.cp.async.wait_group 1;" ::: "memory");
    }
    if constexpr (warp_local_stage)
      __syncwarp();
    else
      __syncthreads();
    MmaAccumulator score[NT];
    MmaOperand kfrag;
#pragma unroll
    for (int t = 0; t < NT; ++t)
      clear_fragment(score[t]);
#pragma unroll
    for (int k = 0; k < kHeadMmaTiles; ++k) {
      const int chunk = (((k % 4) * 16 + b_col) >> 3) ^ lr;
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        ldm_x4(kfrag, &sk[t * 2 + (lu >> 1)][k / 4][lr][chunk << 3]);
        mma_bf16(score[t], qfrag[k], kfrag, score[t]);
      }
    }
    float tile_max[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int tau = (i >> 1) & 1, half = i >> 2;
        const uint32_t limits = entries[t * 2 + half];
        const int visible = (limits >> (4 * (query_offset + tau))) & 15;
        const bool valid = 2 * t0 + (i & 1) < visible;
        score[t].x[i] = valid ? score[t].x[i] : -INFINITY;
        tile_max[tau] = fmaxf(tile_max[tau], score[t].x[i]);
      }
    }
    float rescale[2];
#pragma unroll
    for (int tau = 0; tau < 2; ++tau) {
      const float tile_peak = head_max(tile_max[tau]);
      const float next = fmaxf(maximum[tau], tile_peak);
      rescale[tau] = next == -INFINITY ? 1.f : __expf((maximum[tau] - next) * p.scale);
      maximum[tau] = next;
      denominator[tau] *= rescale[tau];
    }
    MmaOperand prob[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const int tau = (i >> 1) & 1;
        const float offset = maximum[tau] == -INFINITY ? 0.f : maximum[tau];
        const float a = __expf((score[t].x[i] - offset) * p.scale);
        const float b = __expf((score[t].x[i + 1] - offset) * p.scale);
        *reinterpret_cast<__ppu_bfloat162 *>(&prob[t].x[i / 2]) =
            __floats2bfloat162_rn(a, b);
        denominator[tau] += a + b;
      }
    }
    if (!__all_sync(0xffffffffu, rescale[0] == 1.f && rescale[1] == 1.f)) {
#pragma unroll
      for (int j = 0; j < kHeadMmaTiles; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
          out[j].x[i] *= rescale[(i >> 1) & 1];
      }
    }
    if (warp == 0)
      asm volatile("ppu.cp.async.wait_group 0;" ::: "memory");
    if constexpr (warp_local_stage)
      __syncwarp();
    else
      __syncthreads();
    MmaOperand vfrag;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int j = 0; j < kHeadMmaTiles; ++j) {
        const int chunk = (((j % 4) * 16 + a_col) >> 3) ^ lr;
        ldm_x4_trans(vfrag, &sv[t * 2 + (lu & 1)][j / 4][lr][chunk << 3]);
        mma_bf16(out[j], prob[t], vfrag, out[j]);
      }
    }
    if constexpr (warp_local_stage)
      __syncwarp();
    else
      __syncthreads();
  }
  if constexpr (PARTS > 1) {
    // Reuse staging memory for the split merge.
    constexpr int nh = 8 * HT, width = nh * 192;
    const float denom = head_sum(denominator[0]);
    const float local =
        denom > 0 ? maximum[0] * p.scale + __logf(denom) : -INFINITY;
    float *partial = reinterpret_cast<float *>(scratch);
    float *stats = partial + (PARTS - 1) * width;
    __syncthreads();
    if (t0 == 0)
      stats[part * nh + head] = local;
    if (part) {
#pragma unroll
      for (int j = 0; j < kHeadMmaTiles; ++j) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int i = half * 4, d = j * 16 + 2 * t0 + half * 8;
          const float inv = denom > 0 ? 1.f / denom : 0.f;
          *reinterpret_cast<float2 *>(partial + (part - 1) * width + head * 192 +
                                      d) =
              make_float2(out[j].x[i] * inv, out[j].x[i + 1] * inv);
        }
      }
    }
    __syncthreads();
    if (part == 0) {
      float m = -INFINITY, weights[PARTS], sum = 0.f;
#pragma unroll
      for (int t = 0; t < PARTS; ++t)
        m = fmaxf(m, stats[t * nh + head]);
#pragma unroll
      for (int t = 0; t < PARTS; ++t) {
        weights[t] = m == -INFINITY ? 0.f : __expf(stats[t * nh + head] - m);
        sum += weights[t];
      }
      if (t0 == 0)
        p.lse[int64_t(row) * kHeads + head] =
            sum > 0 ? m + __logf(sum) : -INFINITY;
#pragma unroll
      for (int j = 0; j < kHeadMmaTiles; ++j) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int i = half * 4, d = j * 16 + 2 * t0 + half * 8;
          const float w0 = denom > 0 ? weights[0] / denom : 0.f;
          float a = out[j].x[i] * w0, b = out[j].x[i + 1] * w0;
#pragma unroll
          for (int t = 1; t < PARTS; ++t) {
            float2 x = *reinterpret_cast<const float2 *>(
                partial + (t - 1) * width + head * 192 + d);
            a += weights[t] * x.x;
            b += weights[t] * x.y;
          }
          const float inv = sum > 0 ? 1.f / sum : 0.f;
          *reinterpret_cast<__ppu_bfloat162 *>(p.out + row * p.os + head * p.oh +
                                              d) =
              __floats2bfloat162_rn(a * inv, b * inv);
        }
      }
    }
    return;
  }
#pragma unroll
  for (int tau = 0; tau < 2; ++tau) {
    if (row + tau >= end_row)
      continue;
    const int64_t output_row = int64_t(row + tau) * kHeads + head;
    const float denom = head_sum(denominator[tau]);
    const float inv = denom > 0 ? 1.f / denom : 0.f;
    const float lse =
        denom > 0 ? maximum[tau] * p.scale + __logf(denom) : -INFINITY;
    if (t0 == 0) {
      p.lse[output_row] = lse;
    }
#pragma unroll
    for (int j = 0; j < kHeadMmaTiles; ++j) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int i = 2 * tau + 4 * half, d = j * 16 + 2 * t0 + 8 * half;
        const float a = out[j].x[i] * inv, b = out[j].x[i + 1] * inv;
        *reinterpret_cast<__ppu_bfloat162 *>(p.out + (row + tau) * p.os +
                                            head * p.oh + d) =
            __floats2bfloat162_rn(a, b);
      }
    }
  }
}

template <int Heads>
__device__ __forceinline__ __ppu_bfloat16 *paired_cube(__ppu_bfloat16 *base,
                                                     int region, int cube) {
  if constexpr (Heads == 8)
    return base + (region * 3 + cube) * 512;
  // H16 stages the first two 64-dimension cubes together.
  const int offset = cube < 2
      ? region * 1024 + cube * 512
      : 8192 + region * 512;
  return base + offset;
}

template <int Cubes>
__device__ __forceinline__ void stage_region_cubes(
    __ppu_bfloat16 *dst, const __ppu_bfloat16 *src, int extent, int start) {
  const unsigned addr = __cvta_generic_to_shared(dst);
  asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global.2d.tile.LLC::128B.padz.swzl.b16 "
               "[%0], [%1], {%2, %3, %4}, {%2, %5, %4}, {%6, %7}, {%8, %9, %8}, %8;"
               :: "r"(addr), "l"(src), "r"(kCubeWidth), "r"(extent), "r"(Cubes),
                  "r"(kRegionTokens), "l"(kRowBytes), "l"(kCubeRowBytes), "r"(0),
                  "r"(start) : "memory");
}

template <int Heads>
__device__ __forceinline__ void stage_shared_region(
    __ppu_bfloat16 *dst, const __ppu_bfloat16 *src, int region, int extent, int start) {
  if constexpr (Heads == 8) {
    aiu_copy_group(paired_cube<Heads>(dst, region, 0), src, extent, start, kRegionTokens);
  } else {
    stage_region_cubes<2>(paired_cube<Heads>(dst, region, 0), src, extent, start);
    stage_region_cubes<1>(paired_cube<Heads>(dst, region, 2), src + 2 * kCubeWidth,
                          extent, start);
  }
}

template <bool Transpose>
__device__ __forceinline__ void load_paired(MmaOperand &fragment,
                                           const __ppu_bfloat16 *src,
                                           int leading_offset, int secondary_offset) {
  auto *words = fragment.x;
  const uint32_t addr = uint32_t(__cvta_generic_to_shared(src)) / 16;
  if constexpr (Transpose)
    asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16.trans "
                 "{%0, %1, %2, %3}, [%4], %5, %6, %7;"
                 : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]), "=r"(words[3])
                 : "r"(addr), "r"(leading_offset), "r"(secondary_offset), "r"(0));
  else
    asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.b16 "
                 "{%0, %1, %2, %3}, [%4], %5, %6, %7;"
                 : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]), "=r"(words[3])
                 : "r"(addr), "r"(leading_offset), "r"(secondary_offset), "r"(0));
}

template <bool Private, int Heads>
__device__ __forceinline__ void native_tile(const Params &p, int first, int end_row,
                                             int walk, int row_offset = 0) {
  constexpr bool TwoRows = Heads == 8 && !Private;
  constexpr int Halves = Heads == 16 || TwoRows ? 2 : 1;
  constexpr int StageWarps = Heads == 8 ? 4 : 8;
  constexpr int BS = Private ? 16 : 64, TG = BS / 8, NT = BS / 16;
  constexpr int key_batch = !Private && Heads == 8 ? 2 : 1;
  constexpr int value_batch = Private ? 1 : (Heads == 8 ? 2 : kHeadMmaTiles);
  const int warp = __builtin_ppu_to_uniform_b32(threadIdx.x / 32);
  const int lane = threadIdx.x % 32;
  const int row = first + warp + row_offset, head = lane >> 2;
  if constexpr (Private) {
    if (row >= end_row)
      return;
  }
  const int t0 = lane & 3, lr = lane & 7, lu = lane >> 3;
  const int a_col = 8 * (lu >> 1), b_col = 8 * (lu & 1);
  extern __shared__ __align__(128) unsigned char scratch[];
  // Each private warp reuses its K tile for V once QK has consumed it.
  auto *stage = scratch + (Private ? warp * BS * 192 * 2 : 0);
  auto &sk = *reinterpret_cast<KvStage<TG> *>(stage);
  auto &sv = *reinterpret_cast<KvStage<TG> *>(stage + (Private ? 0 : BS * 192 * 2));

  auto *key_base = reinterpret_cast<__ppu_bfloat16 *>(&sk);
  auto *value_base = reinterpret_cast<__ppu_bfloat16 *>(&sv);

  MmaOperand qfrag[kHeadMmaTiles];
  MmaAccumulator out[kHeadMmaTiles];
  if constexpr (!Private) {
    auto *qstage = reinterpret_cast<__ppu_bfloat16 *>(scratch) + warp * 16 * 192;
    if constexpr (Heads == 8) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int query = row + half * StageWarps;
        const bool present = query < end_row;
        const auto *source = p.q + (present ? query : first) * p.qs;
        aiu_copy_group(qstage + half * 8 * 192, source, 8, present ? 0 : 8, 8);
      }
    } else if (row < end_row) {
      aiu_copy_group(qstage, p.q + row * p.qs, 16, 0, 16);
    }
    asm volatile("ppu.cp.async.commit_group;" ::: "memory");
    asm volatile("ppu.cp.async.wait_group 0;" ::: "memory");
    __syncwarp();
#pragma unroll
    for (int k = 0; k < kHeadMmaTiles; ++k) {
      clear_fragment(out[k]);
      if (TwoRows || row < end_row) {
        const int offset = k < 8 ? (k % 4) * 16 + (k / 4) * 8 : Heads * 128 + (k % 4) * 16;
        const int leading = Heads == 8 ? 192 : 64;
        const int secondary = Heads == 8 ? 65 : 129;
        load_paired<false>(qfrag[k], qstage + offset, leading, k < 8 ? secondary - 2 * (k / 4) : 1);
        // The MMA A operand orders row halves before column halves.
        const uint32_t second = qfrag[k].x[1];
        qfrag[k].x[1] = qfrag[k].x[2];
        qfrag[k].x[2] = second;
      } else {
        clear_fragment(qfrag[k]);
      }
    }
    __syncthreads();
  } else {
#pragma unroll
    for (int k = 0; k < kHeadMmaTiles; ++k) {
      clear_fragment(qfrag[k]);
      clear_fragment(out[k]);
#pragma unroll
      for (int tau = 0; tau < Halves; ++tau) {
        const int qhead = head + (Heads == 16 ? tau * 8 : 0);
        const auto *src = p.q + row * p.qs + qhead * p.qh + k * 16 + 2 * t0;
        *reinterpret_cast<__ppu_bfloat162 *>(&qfrag[k].x[tau]) =
            *reinterpret_cast<const __ppu_bfloat162 *>(src);
        *reinterpret_cast<__ppu_bfloat162 *>(&qfrag[k].x[tau + 2]) =
            *reinterpret_cast<const __ppu_bfloat162 *>(src + 8);
      }
    }
  }
  constexpr float log2e = 1.4426950408889634f;
  constexpr float ln2 = 0.6931471805599453f;
  constexpr float initial_max = -FLT_MAX;
  float maximum[2] = {initial_max, initial_max}, denominator[2] = {0, 0};
  const float scale2 = p.scale * log2e;
  auto prefetch_key = [&](int next_base) {
#pragma unroll
    for (int r = 0; r < TG; r += StageWarps) {
      const int slot = warp + r;
      const unsigned tag = __builtin_ppu_to_uniform_b32(next_base + slot < walk
          ? unsigned(p.physical[int64_t(first) * kWidth + next_base + slot]) : 0);
      const int64_t offset = int64_t(tag >> 4) * 1536;
      stage_shared_region<Heads>(key_base, p.k + offset, slot, 8, 0);
    }
    asm volatile("ppu.cp.async.commit_group;" ::: "memory");
  };
  if constexpr (!Private) {
    if (walk)
      prefetch_key(0);
  }
  uint32_t cached_tag = 0, cached_limits[TwoRows ? 2 : 1]{};
  int scan = 0;
  uint32_t remaining = 0, lane_entry = 0;
  for (int base = 0; Private ? (scan < walk || remaining != 0) : base < walk; base += TG) {
    uint64_t entries[Private ? TG : 1];
    uint32_t row_limits[TwoRows ? 2 : 1]{};
    if constexpr (Private) {
      // Consume this row's regions from each 32-entry union window.
#pragma unroll
      for (int r = 0; r < TG; ++r) {
        while (!remaining && scan < walk) {
          const auto entry = scan + lane < walk ? p.read_slot(first, scan + lane) : 0;
          const auto n = (uint32_t(entry) >> (4 * (warp + row_offset))) & 15u;
          lane_entry = (uint32_t(entry >> 32) & ~15u) | n;
          remaining = __ballot_sync(0xffffffffu, n != 0);
          scan += 32;
        }
        entries[r] = 0;
        if (remaining) {
          const uint32_t id = __shfl_sync(0xffffffffu, lane_entry, __ffs(remaining) - 1);
          entries[r] = (uint64_t(id) << 32) | (id & 15u);
          remaining &= remaining - 1;
        }
      }
    } else {
      static_assert(32 % TG == 0);
      // Build four eight-region masks once per metadata load.
      if ((base & 31) == 0) {
        const uint64_t entry = base + lane < walk ? p.read_slot(first, base + lane) : 0;
        cached_tag = uint32_t(entry >> 32);
#pragma unroll
        for (int half = 0; half < (TwoRows ? 2 : 1); ++half) {
          const int shift = 4 * (warp + StageWarps * half);
          uint32_t bits = ((uint32_t(entry) >> shift) & 15u) << (4 * (lane & 7));
#pragma unroll
          for (int step = 1; step < TG; step *= 2)
            bits |= __shfl_xor_sync(0xffffffffu, bits, step);
          cached_limits[half] = bits;
        }
      }
#pragma unroll
      for (int half = 0; half < (TwoRows ? 2 : 1); ++half)
        row_limits[half] = __builtin_ppu_to_uniform_b32(
            __shfl_sync(0xffffffffu, cached_limits[half], base & 31));
    }
    if constexpr (Private) {
#pragma unroll
      for (int r = 0; r < TG; ++r)
        p.copy<8>(&sk[r][0][0][0], entries[r], false);
      asm volatile("ppu.cp.async.commit_group;" ::: "memory");
      asm volatile("ppu.cp.async.wait_group 0;" ::: "memory");
    } else {
#pragma unroll
      for (int r = 0; r < TG; r += StageWarps) {
        const int slot = warp + r;
        const uint32_t tag = __builtin_ppu_to_uniform_b32(
            __shfl_sync(0xffffffffu, cached_tag, (base & 31) + slot));
        const int64_t offset = int64_t(tag >> 4) * 1536;
        const int visible = tag & 15;
        const int start = visible == 0 ? 8 : 0;
        stage_shared_region<Heads>(value_base, p.v + offset, slot, visible + start, start);
      }
      asm volatile("ppu.cp.async.commit_group;" ::: "memory");
      asm volatile("ppu.cp.async.wait_group 1;" ::: "memory");
    }
    if constexpr (Private)
      __syncwarp();
    else
      __syncthreads();
    MmaAccumulator score[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t)
      clear_fragment(score[t]);
#pragma unroll
    for (int k = 0; k < kHeadMmaTiles; ++k) {
      const int chunk = (((k % 4) * 16 + b_col) >> 3) ^ lr;
#pragma unroll
      for (int t = 0; t < NT; t += key_batch) {
        MmaOperand keys[key_batch];
#pragma unroll
        for (int i = 0; i < key_batch; ++i)
          if constexpr (Private)
            ldm_x4(keys[i], &sk[(t + i) * 2 + (lu >> 1)][k / 4][lr][chunk << 3]);
          else
            load_paired<false>(keys[i],
                paired_cube<Heads>(key_base, (t + i) * 2, k < 8 ? 0 : k / 4)
                    + (k % 4) * 16 + (k < 8 ? (k / 4) * 8 : 0),
                Heads == 8 ? 192 : (k < 8 ? 128 : 64),
                k < 8 ? 65 - 2 * (k / 4) : 1);
#pragma unroll
        for (int i = 0; i < key_batch; ++i)
          mma_bf16(score[t + i], qfrag[k], keys[i], score[t + i]);
      }
    }
    if constexpr (Private) {
      __syncwarp();
#pragma unroll
      for (int r = 0; r < TG; ++r)
        p.copy<8>(&sv[r][0][0][0], entries[r], true);
      asm volatile("ppu.cp.async.commit_group;" ::: "memory");
    }
    float tile_max[2] = {-INFINITY, -INFINITY};
    if (!Private && row_limits[0] == 0x88888888u &&
        (!TwoRows || row_limits[TwoRows ? 1 : 0] == 0x88888888u)) {
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
          tile_max[(i >> 1) & 1] = fmaxf(tile_max[(i >> 1) & 1], score[t].x[i]);
      }
    } else {
#pragma unroll
      for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int tau = (i >> 1) & 1, half = i >> 2;
          const int visible = Private ? int(entries[t * 2 + half])
                                    : (row_limits[TwoRows ? tau : 0] >> (4 * (t * 2 + half))) & 15;
          const bool valid = 2 * t0 + (i & 1) < visible;
          score[t].x[i] = valid ? score[t].x[i] : -INFINITY;
          tile_max[tau] = fmaxf(tile_max[tau], score[t].x[i]);
        }
      }
    }
    if (!__all_sync(0xffffffffu, tile_max[0] * scale2 <= maximum[0] &&
                                  tile_max[1] * scale2 <= maximum[1])) {
      float rescale[2];
#pragma unroll
      for (int tau = 0; tau < 2; ++tau) {
        const float tile_peak = head_max(tile_max[tau]);
        const float bound = ceilf(tile_peak * (scale2 * 0.125f)) * 8.f;
        const bool first_nonempty = Heads == 8 && maximum[tau] == initial_max && tile_peak != -INFINITY;
        const float next = first_nonempty ? tile_peak * scale2 : fmaxf(maximum[tau], bound);
        rescale[tau] = __exp2f(maximum[tau] - next);
        maximum[tau] = next;
        denominator[tau] *= rescale[tau];
      }
#pragma unroll
      for (int j = 0; j < kHeadMmaTiles; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
          out[j].x[i] *= rescale[(i >> 1) & 1];
      }
    }
    MmaOperand prob[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int i = 0; i < 8; i += 2) {
        const int tau = (i >> 1) & 1;
        const float a = __exp2f(fmaf(score[t].x[i], scale2, -maximum[tau]));
        const float b = __exp2f(fmaf(score[t].x[i + 1], scale2, -maximum[tau]));
        *reinterpret_cast<__ppu_bfloat162 *>(&prob[t].x[i / 2]) =
            __floats2bfloat162_rn(a, b);
        // Use unrounded probabilities for the FP32 denominator.
        denominator[tau] += a + b;
      }
    }
    asm volatile("ppu.cp.async.wait_group 0;" ::: "memory");
    if constexpr (Private)
      __syncwarp();
    else
      __syncthreads();
    // All warps have consumed K; reuse its buffer while PV reads V.
    if constexpr (!Private) {
      if (base + TG < walk)
        prefetch_key(base + TG);
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
      for (int j = 0; j < kHeadMmaTiles; j += value_batch) {
        MmaOperand values[value_batch];
#pragma unroll
        for (int i = 0; i < value_batch; ++i) {
          const int d = j + i;
          const int chunk = (((d % 4) * 16 + a_col) >> 3) ^ lr;
          if constexpr (Private)
            ldm_x4_trans(values[i], &sv[t * 2 + (lu & 1)][d / 4][lr][chunk << 3]);
          else
            load_paired<true>(values[i],
                paired_cube<Heads>(value_base, t * 2, d < 8 ? 0 : d / 4)
                    + (d % 4) * 16 + (d < 8 ? (d / 4) * 8 : 0),
                d < 8 ? 64 : 1, Heads == 8 ? 192 : (d < 8 ? 128 : 64));
        }
#pragma unroll
        for (int i = 0; i < value_batch; ++i)
          mma_bf16(out[j + i], prob[t], values[i], out[j + i]);
      }
    }
    if constexpr (Private)
      __syncwarp();
    else
      __syncthreads();
  }
#pragma unroll
  for (int tau = 0; tau < Halves; ++tau) {
    const int qrow = row + (TwoRows ? StageWarps * tau : 0);
    if (qrow >= end_row)
      continue;
    const int h = head + (Heads == 16 ? tau * 8 : 0);
    const int64_t output_row = int64_t(qrow) * kHeads + h;
    const float denom = head_sum(denominator[tau]);
    const float inv = denom > 0 ? 1.f / denom : 0.f;
    const float lse =
        denom > 0 ? maximum[tau] * ln2 + __logf(denom) : -INFINITY;
    if (t0 == 0) {
      p.lse[output_row] = lse;
    }
#pragma unroll
    for (int j = 0; j < kHeadMmaTiles; ++j) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int i = 2 * tau + 4 * half;
        const int d = !Private && j < 8
            ? (j % 4) * 16 + (j / 4) * 8 + half * 64 + 2 * t0
            : j * 16 + 2 * t0 + 8 * half;
        const float a = out[j].x[i] * inv, b = out[j].x[i + 1] * inv;
        *reinterpret_cast<__ppu_bfloat162 *>(p.out + qrow * p.os + h * p.oh + d) =
            __floats2bfloat162_rn(a, b);
      }
    }
  }
}

template <int Heads>
__global__ __launch_bounds__(Heads * 16) void step5_native_attention_kernel(Params p) {
  constexpr int tiles = kGroupRows / 8;
  const int group = p.groups - 1 - (Heads == 16 ? blockIdx.x : blockIdx.x / tiles);
  const int end_row = min(p.rows, p.ends[group]);
  const int start_tile = Heads == 16 ? 0 : blockIdx.x % tiles;
  const int end_tile = Heads == 16 ? tiles : start_tile + 1;
  for (int tile = start_tile; tile < end_tile; ++tile) {
    const int first = p.starts[group] + 8 * tile;
    if (first >= end_row)
      break;
    if (Heads == 16 && tile)
      __syncthreads();
    const int count = p.tile_counts[p.tile_index(group, tile)];
    if (count > 4 * kWidth) {
      if constexpr (Heads == 8) {
#pragma unroll 1
        for (int offset = 0; offset < 8; offset += 4)
          native_tile<true, 8>(p, first, end_row, count, offset);
      } else {
        native_tile<true, 16>(p, first, end_row, count);
      }
    } else {
      native_tile<false, Heads>(p, first, end_row, count);
    }
  }
}

template <int BQ>
void launch(Params p, hggcStream_t stream) {
  constexpr int HT = kHeads / 8;
  constexpr int BS = BQ == 1 ? 16 : 64;
  constexpr int PARTS = BQ == 1 ? 8 : 1;
  constexpr bool group_cta = BQ == 8 && kHeads == 16;
  const int blocks = p.groups * (group_cta ? 1 : kGroupRows / BQ);
  if constexpr (BQ > 1) {
    constexpr int bytes = (kWindowRegions + BQ * kWords) * 4;
    if (p.capacity > kWindowRegions) {
      host::RuntimeDeviceCheck(hggcFuncSetAttribute(
          step5_prepare_union_kernel<BQ, group_cta, 2 * kWords>,
          hggcFuncAttributeMaxDynamicSharedMemorySize, 2 * bytes));
      step5_prepare_union_kernel<BQ, group_cta, 2 * kWords>
          <<<blocks, 2 * kWords, 2 * bytes, stream>>>(p);
    } else {
      step5_prepare_union_kernel<BQ, group_cta, kWords>
          <<<blocks, kWords, bytes, stream>>>(p);
    }
    host::RuntimeDeviceCheck();
  }
  constexpr int threads = (BQ == 1 ? 1 : BQ / 2) * HT * 32;
  constexpr int bytes = BS * 192 * 4 * PARTS;
  static_assert(PARTS == 1 ||
                ((PARTS - 1) * HT * 8 * 192 + PARTS * HT * 8) * 4 <= bytes);
  if constexpr (BQ == 8)
    step5_native_attention_kernel<kHeads><<<blocks, threads, bytes, stream>>>(p);
  else
    step5_paged_attention_kernel<BQ, HT, BS, PARTS>
        <<<blocks, threads * PARTS, bytes, stream>>>(p);
}

void run(
    tvm::ffi::TensorView q, tvm::ffi::TensorView k, tvm::ffi::TensorView v,
    tvm::ffi::TensorView packed, tvm::ffi::TensorView counts, tvm::ffi::TensorView positions,
    tvm::ffi::TensorView starts, tvm::ffi::TensorView ends, tvm::ffi::TensorView physical,
    tvm::ffi::TensorView limits, tvm::ffi::TensorView tile_counts, tvm::ffi::TensorView out,
    tvm::ffi::TensorView lse, double scale) {
  using namespace host;
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  auto rows = SymbolicSize{"rows"};
  auto slots = SymbolicSize{"KV slots"};
  auto groups = SymbolicSize{"groups"};
  auto capacity = SymbolicSize{"tile count capacity"};
  TensorMatcher({rows, kHeads, kHeadDim})
      .with_dtype<__ppu_bfloat16>().with_device(device).ensure_alignment(16).verify(q).verify(out);
  TensorMatcher({slots, 1, kHeadDim})
      .with_dtype<__ppu_bfloat16>().with_device(device).ensure_alignment(16).verify(k).verify(v);
  TensorMatcher({rows, kWidth}).with_dtype<int64_t>().with_device(device).verify(packed);
  TensorMatcher({rows, kWidth}).with_dtype<int32_t>().with_device(device).verify(physical).verify(limits);
  TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(counts).verify(positions);
  TensorMatcher({groups}).with_dtype<int32_t>().with_device(device).verify(starts).verify(ends);
  TensorMatcher({groups, capacity}).with_dtype<int32_t>().with_device(device).verify(tile_counts);
  TensorMatcher({rows, kHeads}).with_dtype<float>().with_device(device).verify(lse);
  const auto n = rows.unwrap(), s = slots.unwrap(), g = groups.unwrap();
  RuntimeCheck(n <= INT32_MAX && g <= INT32_MAX / kGroupRows &&
               s % 64 == 0 && s <= (int64_t(1) << 31));
  RuntimeCheck(std::isfinite(scale) && scale > 0);
  if (n == 0 || g == 0) return;

  Params p{};
  p.q = static_cast<const __ppu_bfloat16 *>(q.data_ptr());
  p.k = static_cast<const __ppu_bfloat16 *>(k.data_ptr());
  p.v = static_cast<const __ppu_bfloat16 *>(v.data_ptr());
  p.out = static_cast<__ppu_bfloat16 *>(out.data_ptr());
  p.lse = static_cast<float *>(lse.data_ptr());
  p.live = static_cast<const int32_t *>(counts.data_ptr());
  p.pos = static_cast<const int32_t *>(positions.data_ptr());
  p.packed = static_cast<const int64_t *>(packed.data_ptr());
  p.starts = static_cast<const int32_t *>(starts.data_ptr());
  p.ends = static_cast<const int32_t *>(ends.data_ptr());
  p.physical = static_cast<int32_t *>(physical.data_ptr());
  p.limits = static_cast<int32_t *>(limits.data_ptr());
  p.tile_counts = static_cast<int32_t *>(tile_counts.data_ptr());
  p.capacity = capacity.unwrap();
  p.rows = static_cast<int>(n);
  p.groups = static_cast<int>(g);
  p.has_kv = s != 0;
  p.scale = static_cast<float>(scale);
  p.qs = q.stride(0);
  p.qh = q.stride(1);
  p.os = out.stride(0);
  p.oh = out.stride(1);

  const auto target = device.unwrap();
  const tvm::ffi::CUDADeviceGuard guard(target.device_id);
  const auto stream = LaunchKernel::resolve_device(target);
  int tile = n <= 32 ? 1 : (n <= 256 ? 4 : 8);
  if (tile > 1 && p.capacity < kGroupRows / tile)
    tile = 1;
  switch (tile) {
  case 1:
    launch<1>(p, stream);
    break;
  case 4:
    launch<4>(p, stream);
    break;
  case 8:
    launch<8>(p, stream);
    break;
  }
  RuntimeDeviceCheck();
}

} // namespace sglang::step5_prefill
