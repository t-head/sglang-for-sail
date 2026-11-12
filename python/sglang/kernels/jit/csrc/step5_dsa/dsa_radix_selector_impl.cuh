// SPDX-License-Identifier: Apache-2.0
//
// C++ implementation of the Step5 DSA fixed-decode top-k region selector.
// The streaming radix dataflow keeps shared memory independent of the context
// length, so one kernel covers short contexts and the 1M-token target.
//
// Radix scheme: [11, 11, 10] (3 rounds) replacing the original [8,8,8,8].
// Rounds 0-1 use 2048-bin histograms (11-bit extraction); round 2 uses
// 1024 bins (10-bit).  Only standard CUDA APIs are used.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

// Each include of this file yields one thread-count specialization; the
// namespace encodes the thread count so two specializations coexist in one
// translation unit without symbol collisions. There is deliberately no
// include guard: the JIT entry header defines DSA_SELECTOR_THREADS and
// includes this file once per specialization.
#ifndef DSA_SELECTOR_THREADS
#error "define DSA_SELECTOR_THREADS (256 or 1024) before including this file"
#endif

#if DSA_SELECTOR_THREADS == 256
namespace sglang {
namespace dsa_selector_w256 {
#elif DSA_SELECTOR_THREADS == 1024
namespace sglang {
namespace dsa_selector_w1024 {
#else
#error "DSA_SELECTOR_THREADS must be 256 or 1024"
#endif

constexpr int kThreads = DSA_SELECTOR_THREADS;
constexpr int kRadix = 1024;              // 2^10 for last round
constexpr int kCoarseRadix = 2048;        // 2^11 for first two rounds
constexpr int kNumSelectorRounds = 3;     // [11, 11, 10]
constexpr int kStreamingChunkRegions = 16384;
constexpr float kRegionEpsilon = 1.0e-9f;

// Hybrid score cache for the single-CTA selector path.
// We keep the total cache at 8192 floats (r <= 8192) but split it between
// registers and shared memory to reduce smem footprint and improve occupancy.
// The per-thread register budget is tuned per warp size so that both 256 and
// 1024 variants stay within the SM register file and keep a non-zero smem
// buffer (zero-length arrays are avoided).
//   - warp-256: 8 regs/thread  -> 2048 register + 6144 smem
//   - warp-1024: 4 regs/thread -> 4096 register + 4096 smem
constexpr int kRegisterCachePerThread = (kThreads == 1024) ? 4 : 8;
constexpr int kRegisterCacheRegions = kRegisterCachePerThread * kThreads;
constexpr int kSmemBufferRegions = 8192 - kRegisterCacheRegions;
constexpr int kMaxBufferedRegions = 8192;  // register + smem cache capacity
static_assert(kRegisterCacheRegions <= kMaxBufferedRegions,
              "Register cache cannot exceed total buffered regions");
static_assert(kSmemBufferRegions > 0,
              "Smem buffer must be positive for all supported warp sizes");

static_assert(kThreads >= 256 && kThreads <= 1024 &&
                  (kThreads & (kThreads - 1)) == 0,
              "Step5 DSA selector threads must be a power of two in [256,1024]");

// Shared storage for selector and chunk kernels.
// The histogram and (equal_counts + warp_sum) are used in disjoint phases,
// so they share memory via a union.  The histogram is sized for the coarsest
// round (kCoarseRadix = 2048); equal_counts needs kThreads entries and
// warp_sum needs kThreads/32, both smaller than 2048.
struct DsaRadixShared {
  union {
    int histogram[kCoarseRadix];
    struct {
      int equal_counts[kThreads];
      int warp_sum[kThreads / 32];
    };
  };
  uint32_t prefix_value;
  uint32_t prefix_mask;
  int threshold;
  int remain;
  int selected_base;
  int output_counter;
  int count_greater;
  // Hybrid score cache for the single-CTA selector path.
  // Regions [0, kRegisterCacheRegions) are held in per-thread registers;
  // regions [kRegisterCacheRegions, kMaxBufferedRegions) are cached here.
  float score_buffer[kSmemBufferRegions];
};

struct DsaMergeRadixShared {
  union {
    int histogram[kCoarseRadix];
    int warp_sum[kThreads / 32];
  };
  uint32_t score_prefix_value;
  uint32_t score_prefix_mask;
  uint32_t region_prefix_value;
  uint32_t region_prefix_mask;
  int threshold;
  int remain;
  int selected_base;
  int output_counter;
  int count_greater;
};

__device__ __forceinline__ uint32_t ordered_score(float score, int region) {
  float const encoded = score + float(region) * kRegionEpsilon;
  uint32_t const raw = __float_as_uint(encoded);
  return raw ^ ((raw >> 31) != 0 ? 0xffffffffu : 0x80000000u);
}

__device__ __forceinline__ void store_region(
    int region, int slot, int request, int head, int heads, int output_width,
    int const* block_table, int block_table_stride, int block_table_length,
    int regions_per_page, int region_block, bool packed, int* region_output,
    int* start_output) {
  int selected = region;
  int start = region * region_block;
  if (packed) {
    int const logical_page = region / regions_per_page;
    int const fragment = region - logical_page * regions_per_page;
    int const physical_page =
        logical_page >= 0 && logical_page < block_table_length
            ? block_table[request * block_table_stride + logical_page]
            : -1;
    bool const valid = physical_page >= 0;
    selected = valid ? physical_page * regions_per_page + fragment : -1;
    start = valid ? start : -1;
  }
  int const offset = (request * heads + head) * output_width + slot;
  region_output[offset] = selected;
  start_output[offset] = start;
}

// Per-thread staging size for the scatter phase.  Each thread buffers up to
// this many selected region indices in registers; when the buffer fills, the
// warp performs a single batched atomicAdd and flushes the buffered regions
// to contiguous output slots, reducing atomic contention and improving
// coalescing of the output writes.
constexpr int kThreadStaging = 8;

// Compute the contiguous output slot range for this warp's staged selected
// regions and claim it with a single global atomicAdd.  The per-thread counts
// are prefix-summed via warp shuffle (always executed with the full 32-lane
// mask so the SIMT mask stays consistent on PPU).  Returns the base slot
// claimed for the warp and writes this lane's exclusive offset into
// warp_exclusive.
__device__ __forceinline__ int claim_warp_output_slots(
    int staged_count, int thread, DsaRadixShared& storage,
    int& warp_exclusive) {
  int const lane_id = thread & 31;
  int warp_inclusive = staged_count;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    int const n = __shfl_up_sync(0xFFFFFFFF, warp_inclusive, offset);
    if (lane_id >= offset) warp_inclusive += n;
  }
  int const warp_total = __shfl_sync(0xFFFFFFFF, warp_inclusive, 31);
  warp_exclusive = warp_inclusive - staged_count;
  int base_slot = 0;
  if (lane_id == 0) {
    base_slot = atomicAdd(&storage.output_counter, warp_total);
  }
  base_slot = __shfl_sync(0xFFFFFFFF, base_slot, 0);
  return base_slot;
}

// Hierarchical warp-shuffle suffix sum for equal-count reduction.
// Returns the number of "equal" regions held by threads with a larger global
// index than this thread.  Replaces the tree reduction over
// storage.equal_counts[] with two warp-level shuffle passes plus two
// __syncthreads, cutting sync overhead from 2*log2(kThreads) to 2.
__device__ __forceinline__ int equal_count_higher_equal(
    int local_equal, int thread, DsaRadixShared& storage) {
  int const lane_id = thread & 31;
  int const warp_id = thread >> 5;

  // Step 1: Inclusive suffix sum within each warp.
  int warp_suffix = local_equal;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    int const n = __shfl_down_sync(0xFFFFFFFF, warp_suffix, offset);
    if (lane_id + offset < 32) warp_suffix += n;
  }
  int const warp_total = __shfl_sync(0xFFFFFFFF, warp_suffix, 0);
  if (lane_id == 0) {
    storage.warp_sum[warp_id] = warp_total;
  }
  __syncthreads();

  // Step 2: Suffix sum of per-warp totals (one warp's worth of data).
  if (thread < 32) {
    int const val = (thread < kThreads / 32) ? storage.warp_sum[thread] : 0;
    int suffix = val;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      int const n = __shfl_down_sync(0xFFFFFFFF, suffix, offset);
      if (lane_id + offset < 32) suffix += n;
    }
    if (thread < kThreads / 32) {
      storage.warp_sum[thread] = suffix;
    }
  }
  __syncthreads();

  // Step 3: Combine intra-warp suffix (excluding this lane) with suffix of
  // all later warps.
  return (warp_suffix - local_equal) +
         ((warp_id + 1 < kThreads / 32) ? storage.warp_sum[warp_id + 1] : 0);
}

// Warp-based threshold bin selection parameterised on histogram size.
//
// Phase 1: Warp-level inclusive prefix sum via __shfl_up_sync; warp totals
//          published to shared memory.
// Phase 2: Warp-0 computes the serial exclusive prefix of warp totals and
//          publishes the total (via output_counter) and per-warp exclusive
//          offsets back to warp_sum[].
// Phase 3: All bin-owning threads combine the warp exclusive offset with
//          their intra-warp prefix to obtain the per-bin exclusive prefix
//          sum, then search for the threshold in parallel.
template <int n_bins, typename SharedStorage>
__device__ __forceinline__ void choose_threshold_parallel(
    SharedStorage& storage, int thread) {
  constexpr int kWarps = kThreads / 32;
  constexpr int kOwningThreads = (kThreads < n_bins) ? kThreads : n_bins;
  constexpr int kItemsPerThread =
      (kThreads <= n_bins) ? (n_bins / kThreads) : 0;

  int const lane_id = thread & 31;
  int const warp_id = thread >> 5;
  bool const owns_bin = (thread < kOwningThreads);

  // Step 1: Load owned histogram items into registers.
  // Max items per thread occurs at kThreads=256, n_bins=4096 => 16.
  int local_sum = 0;
  int orig[16];
  static_assert(kThreads >= 256, "orig[] sized for kThreads >= 256");
  if constexpr (kThreads <= n_bins) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      orig[i] = storage.histogram[thread * kItemsPerThread + i];
      local_sum += orig[i];
    }
  } else {
    orig[0] = owns_bin ? storage.histogram[thread] : 0;
    local_sum = orig[0];
  }
  __syncthreads();

  // Step 2: Warp-level inclusive prefix sum via __shfl_up_sync.
  int warp_inc = local_sum;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    int n = __shfl_up_sync(0xFFFFFFFF, warp_inc, offset);
    if (lane_id >= offset) warp_inc += n;
  }
  int const warp_exc = warp_inc - local_sum;

  // Step 3: Last lane of each warp publishes the warp total.
  if (lane_id == 31) {
    storage.warp_sum[warp_id] = warp_inc;
  }
  __syncthreads();

  // Step 4: Warp 0 computes the exclusive prefix of warp totals in parallel
  // using warp shuffle. Each lane loads one warp total, performs a Kogge-Stone
  // inclusive prefix sum, then writes back the exclusive value. This reduces
  // the serial latency of the original loop and helps warps arrive at the next
  // barrier more closely together.
  if (thread < 32) {
    int const val = (lane_id < kWarps) ? storage.warp_sum[lane_id] : 0;
    int prefix = val;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      int const n = __shfl_up_sync(0xFFFFFFFF, prefix, offset);
      if (lane_id >= offset) prefix += n;
    }
    int const exclusive = prefix - val;
    if (lane_id < kWarps) {
      storage.warp_sum[lane_id] = exclusive;
    }
    if (lane_id == kWarps - 1) {
      storage.output_counter = prefix;  // total
      storage.remain = min(storage.remain, prefix);
    }
  }
  __syncthreads();

  // Step 5: Read total and per-warp exclusive prefix from shared memory.
  int const total = storage.output_counter;
  int const warp_exclusive = owns_bin ? storage.warp_sum[warp_id] : 0;

  // Step 6: Combine warp_exclusive + warp_exc for per-bin exclusive prefix.
  int prefix_sum = warp_exclusive + warp_exc;
  int local_threshold = -1;
  int local_count_greater = 0;

  if constexpr (kThreads <= n_bins) {
#pragma unroll
    for (int i = 0; i < kItemsPerThread; ++i) {
      prefix_sum += orig[i];
      int const count_greater = total - prefix_sum;
      if (count_greater < storage.remain &&
          count_greater + orig[i] >= storage.remain) {
        local_threshold = thread * kItemsPerThread + i;
        local_count_greater = count_greater;
      }
    }
  } else {
    if (owns_bin) {
      prefix_sum += orig[0];
      int const count_greater = total - prefix_sum;
      if (count_greater < storage.remain &&
          count_greater + orig[0] >= storage.remain) {
        local_threshold = thread;
        local_count_greater = count_greater;
      }
    }
  }

  // Step 7: Publish result.
  if (local_threshold >= 0) {
    storage.threshold = local_threshold;
    storage.count_greater = local_count_greater;
  }
  __syncthreads();
}

// Selector kernel: 3 rounds [11, 11, 10].
__global__ void dsa_radix_selector_kernel(
    float const* scores, int64_t const* query_positions,
    int const* block_table, int* region_output, int* start_output, int heads,
    int regions, int topk, int region_block, int page_size,
    int block_table_stride, int block_table_length, bool packed) {
  extern __shared__ char raw_shared[];
  auto& storage = *reinterpret_cast<DsaRadixShared*>(raw_shared);
  int const work = blockIdx.x;
  int const request = work / heads;
  int const head = work - request * heads;
  int const thread = threadIdx.x;
  int const output_width = topk + 1;
  int const current = min(
      max(int(query_positions[request]), 0) / region_block, regions - 1);
  int const target = min(topk, current);
  float const* score_row = scores + (request * heads + head) * regions;

  for (int slot = thread; slot < output_width; slot += kThreads) {
    int const offset = work * output_width + slot;
    region_output[offset] = -1;
    start_output[offset] = -1;
  }
  if (thread == 0) {
    storage.prefix_value = 0;
    storage.prefix_mask = 0;
    storage.remain = target;
    storage.selected_base = 0;
  }
  __syncthreads();

  // Layer 1: hybrid score cache for small contexts.
  // Split the 8192-element cache between registers (2048 floats) and shared
  // memory (6144 floats) to reduce smem footprint while retaining the benefits
  // of a one-time load.  Regions >= kMaxBufferedRegions fall back to streaming
  // from global memory.
  bool const use_hybrid_cache = (current <= kMaxBufferedRegions);
  int const register_count = use_hybrid_cache ? min(current, kRegisterCacheRegions) : 0;
  int const smem_count = use_hybrid_cache ? max(0, current - kRegisterCacheRegions) : 0;
  float register_cache[kRegisterCachePerThread];
#pragma unroll
  for (int i = 0; i < kRegisterCachePerThread; ++i) {
    register_cache[i] = 0.0f;
  }

  if (register_count > 0) {
#pragma unroll
    for (int i = 0; i < kRegisterCachePerThread; ++i) {
      int const region = thread * kRegisterCachePerThread + i;
      if (region < register_count) {
        register_cache[i] = score_row[region];
      }
    }
  }
  if (smem_count > 0) {
    // Simple scalar load path (instead of cp.async) to verify correctness
    // independent of async copy alignment / behavior.
    for (int smem_region = thread; smem_region < smem_count;
         smem_region += kThreads) {
      int const global_region = kRegisterCacheRegions + smem_region;
      storage.score_buffer[smem_region] = score_row[global_region];
    }
    __syncthreads();
  }
  float const* score_src = use_hybrid_cache ? nullptr : score_row;

  // [11, 11, 10]: shifts = {21, 10, 0}, bins = {2048, 2048, 1024}
#pragma unroll
  for (int round = 0; round < kNumSelectorRounds; ++round) {
    int const n_bins = (round < 2) ? kCoarseRadix : kRadix;
    int const bits   = (round < 2) ? 11 : 10;
    int const shift  = (round < 2) ? (32 - (round + 1) * 11) : 0;
    uint32_t const bin_mask = (1u << bits) - 1u;

    for (int bin = thread; bin < n_bins; bin += kThreads) {
      storage.histogram[bin] = 0;
    }
    __syncthreads();

    uint32_t const prefix_value = storage.prefix_value;
    uint32_t const prefix_mask = storage.prefix_mask;
    if (use_hybrid_cache) {
      // Histogram from register cache (8 elements per thread, unrolled).
#pragma unroll
      for (int i = 0; i < kRegisterCachePerThread; ++i) {
        int const region = thread * kRegisterCachePerThread + i;
        if (region < register_count) {
          uint32_t const key = ordered_score(register_cache[i], region);
          if ((key & prefix_mask) == prefix_value) {
            atomicAdd(&storage.histogram[int((key >> shift) & bin_mask)], 1);
          }
        }
      }
      // Histogram from smem cache (strided).
      for (int smem_region = thread; smem_region < smem_count;
           smem_region += kThreads) {
        int const region = kRegisterCacheRegions + smem_region;
        uint32_t const key = ordered_score(
            storage.score_buffer[smem_region], region);
        if ((key & prefix_mask) == prefix_value) {
          atomicAdd(&storage.histogram[int((key >> shift) & bin_mask)], 1);
        }
      }
    } else {
      for (int region = thread; region < current; region += kThreads) {
        uint32_t const key = ordered_score(score_src[region], region);
        if ((key & prefix_mask) == prefix_value) {
          atomicAdd(&storage.histogram[int((key >> shift) & bin_mask)], 1);
        }
      }
    }
    __syncthreads();

    if (n_bins == kCoarseRadix) {
      choose_threshold_parallel<kCoarseRadix>(storage, thread);
    } else {
      choose_threshold_parallel<kRadix>(storage, thread);
    }

    if (thread == 0 && storage.remain > 0) {
      uint32_t const round_mask = bin_mask << shift;
      storage.prefix_mask |= round_mask;
      storage.prefix_value =
          (storage.prefix_value & ~round_mask) |
          (uint32_t(storage.threshold) << shift);
      storage.selected_base += storage.count_greater;
      storage.remain -= storage.count_greater;
    }
    __syncthreads();
  }

  // Scatter phase: emit selected regions with warp-local staging.
  uint32_t const threshold_key = storage.prefix_value;
  int local_equal = 0;
  if (thread == 0) {
    storage.output_counter = 0;
  }
  __syncthreads();

  int staged_regions[kThreadStaging];
  int staged_count = 0;

  auto process_region_for_scatter = [&](int region, float raw, bool valid) {
    uint32_t key = 0;
    if (valid) {
      key = ordered_score(raw, region);
    }
    bool const greater = valid && (key > threshold_key);
    bool const equal = valid && (key == threshold_key);
    if (greater) {
      staged_regions[staged_count++] = region;
    } else if (equal) {
      ++local_equal;
    }
    if (__any_sync(0xFFFFFFFF, staged_count == kThreadStaging)) {
      int warp_exclusive = 0;
      int const base_slot =
          claim_warp_output_slots(staged_count, thread, storage,
                                  warp_exclusive);
#pragma unroll
      for (int i = 0; i < kThreadStaging; ++i) {
        if (i < staged_count) {
          int const slot = base_slot + warp_exclusive + i;
          if (slot < storage.selected_base) {
            store_region(staged_regions[i], slot, request, head, heads,
                         output_width, block_table, block_table_stride,
                         block_table_length, page_size / region_block,
                         region_block, packed, region_output, start_output);
          }
        }
      }
      staged_count = 0;
    }
  };

  auto flush_staged = [&]() {
    int warp_exclusive = 0;
    int const base_slot =
        claim_warp_output_slots(staged_count, thread, storage, warp_exclusive);
#pragma unroll
    for (int i = 0; i < kThreadStaging; ++i) {
      if (i < staged_count) {
        int const slot = base_slot + warp_exclusive + i;
        if (slot < storage.selected_base) {
          store_region(staged_regions[i], slot, request, head, heads,
                       output_width, block_table, block_table_stride,
                       block_table_length, page_size / region_block,
                       region_block, packed, region_output, start_output);
        }
      }
    }
  };

  if (use_hybrid_cache) {
    // Phase A: scatter from register cache (8 fixed iterations per thread).
    // Always call process_region_for_scatter to keep __any_sync convergent.
#pragma unroll
    for (int i = 0; i < kRegisterCachePerThread; ++i) {
      int const region = thread * kRegisterCachePerThread + i;
      bool const valid = (region < register_count);
      process_region_for_scatter(region, register_cache[i], valid);
    }
    // Flush before switching to smem to keep staged_count bounded.
    flush_staged();
    staged_count = 0;

    // Phase B: scatter from smem cache (fixed iterations to keep warp
    // convergent; invalid lanes simply pass valid=false).
    if (smem_count > 0) {
      int const smem_iters = (smem_count + kThreads - 1) / kThreads;
#pragma unroll
      for (int iter = 0; iter < smem_iters; ++iter) {
        int const smem_region = thread + iter * kThreads;
        bool const valid = (smem_region < smem_count);
        int const region = kRegisterCacheRegions + smem_region;
        float const raw =
            valid ? storage.score_buffer[smem_region] : 0.0f;
        process_region_for_scatter(region, raw, valid);
      }
    }
  } else {
    int const begin = int((int64_t(current) * thread) / kThreads);
    int const end = int((int64_t(current) * (thread + 1)) / kThreads);
    int const iterations = (current + kThreads - 1) / kThreads;
    for (int iter = 0; iter < iterations; ++iter) {
      int const region = begin + iter;
      bool const valid = region < end;
      float const raw = valid ? score_src[region] : 0.0f;
      process_region_for_scatter(region, raw, valid);
    }
  }

  flush_staged();
  int const higher_equal =
      equal_count_higher_equal(local_equal, thread, storage);
  int const local_take =
      min(local_equal, max(storage.remain - higher_equal, 0));
  int emitted = 0;

  if (use_hybrid_cache) {
    // Tie-breaking from register cache (reverse order, deterministic).
    for (int i = kRegisterCachePerThread - 1; i >= 0 && emitted < local_take;
         --i) {
      int const region = thread * kRegisterCachePerThread + i;
      if (region < register_count &&
          ordered_score(register_cache[i], region) == threshold_key) {
        int const slot = storage.selected_base + higher_equal + emitted;
        if (slot < topk) {
          store_region(region, slot, request, head, heads, output_width,
                       block_table, block_table_stride, block_table_length,
                       page_size / region_block, region_block, packed,
                       region_output, start_output);
        }
        ++emitted;
      }
    }
    // Tie-breaking from smem cache (reverse strided order, matching the
    // scatter partition so local_take is consumed from the same regions that
    // contributed to local_equal).
    if (thread < smem_count) {
      int const max_n = (smem_count - 1 - thread) / kThreads;
      for (int n = max_n; n >= 0 && emitted < local_take; --n) {
        int const smem_region = thread + n * kThreads;
        int const region = kRegisterCacheRegions + smem_region;
        if (ordered_score(storage.score_buffer[smem_region], region) ==
            threshold_key) {
          int const slot = storage.selected_base + higher_equal + emitted;
          if (slot < topk) {
            store_region(region, slot, request, head, heads, output_width,
                         block_table, block_table_stride, block_table_length,
                         page_size / region_block, region_block, packed,
                         region_output, start_output);
          }
          ++emitted;
        }
      }
    }
  } else {
    int const begin = int((int64_t(current) * thread) / kThreads);
    int const end = int((int64_t(current) * (thread + 1)) / kThreads);
    for (int region = end - 1; region >= begin && emitted < local_take;
         --region) {
      if (ordered_score(score_src[region], region) == threshold_key) {
        int const slot = storage.selected_base + higher_equal + emitted;
        if (slot < topk) {
          store_region(region, slot, request, head, heads, output_width,
                       block_table, block_table_stride, block_table_length,
                       page_size / region_block, region_block, packed,
                       region_output, start_output);
        }
        ++emitted;
      }
    }
  }

  if (thread == 0) {
    store_region(current, topk, request, head, heads, output_width,
                 block_table, block_table_stride, block_table_length,
                 page_size / region_block, region_block, packed,
                 region_output, start_output);
  }
}

// Chunk kernel: 3 rounds [11, 11, 10] (same as selector).
__global__ void dsa_radix_chunk_kernel(
    float const* scores, int64_t const* query_positions,
    int* candidate_output, int heads, int regions, int topk, int region_block,
    int num_chunks, int candidate_stride) {
  extern __shared__ char raw_shared[];
  auto& storage = *reinterpret_cast<DsaRadixShared*>(raw_shared);
  int const chunk_work = blockIdx.x;
  int const work = chunk_work / num_chunks;
  int const chunk = chunk_work - work * num_chunks;
  int const request = work / heads;
  int const head = work - request * heads;
  int const thread = threadIdx.x;
  int const current = min(
      max(int(query_positions[request]), 0) / region_block, regions - 1);
  int const chunk_begin = chunk * kStreamingChunkRegions;
  int const chunk_end = min(chunk_begin + kStreamingChunkRegions, current);
  int const valid_length = max(chunk_end - chunk_begin, 0);
  int const target = min(topk, valid_length);
  float const* score_row = scores + work * regions;
  int* output_row =
      candidate_output + work * candidate_stride + chunk * topk;

  for (int slot = thread; slot < topk; slot += kThreads) {
    output_row[slot] = -1;
  }
  if (thread == 0) {
    storage.prefix_value = 0;
    storage.prefix_mask = 0;
    storage.remain = target;
    storage.selected_base = 0;
  }
  __syncthreads();

#pragma unroll
  for (int round = 0; round < kNumSelectorRounds; ++round) {
    int const n_bins = (round < 2) ? kCoarseRadix : kRadix;
    int const bits   = (round < 2) ? 11 : 10;
    int const shift  = (round < 2) ? (32 - (round + 1) * 11) : 0;
    uint32_t const bin_mask = (1u << bits) - 1u;

    for (int bin = thread; bin < n_bins; bin += kThreads) {
      storage.histogram[bin] = 0;
    }
    __syncthreads();

    uint32_t const prefix_value = storage.prefix_value;
    uint32_t const prefix_mask = storage.prefix_mask;
    for (int region = chunk_begin + thread; region < chunk_end;
         region += kThreads) {
      uint32_t const key = ordered_score(score_row[region], region);
      if ((key & prefix_mask) == prefix_value) {
        atomicAdd(&storage.histogram[int((key >> shift) & bin_mask)], 1);
      }
    }
    __syncthreads();

    if (n_bins == kCoarseRadix) {
      choose_threshold_parallel<kCoarseRadix>(storage, thread);
    } else {
      choose_threshold_parallel<kRadix>(storage, thread);
    }

    if (thread == 0 && storage.remain > 0) {
      uint32_t const round_mask = bin_mask << shift;
      storage.prefix_mask |= round_mask;
      storage.prefix_value =
          (storage.prefix_value & ~round_mask) |
          (uint32_t(storage.threshold) << shift);
      storage.selected_base += storage.count_greater;
      storage.remain -= storage.count_greater;
    }
    __syncthreads();
  }

  int const begin =
      chunk_begin + int((int64_t(valid_length) * thread) / kThreads);
  int const end =
      chunk_begin + int((int64_t(valid_length) * (thread + 1)) / kThreads);
  uint32_t const threshold_key = storage.prefix_value;
  int local_equal = 0;
  if (thread == 0) {
    storage.output_counter = 0;
  }
  __syncthreads();

  int staged_regions[kThreadStaging];
  int staged_count = 0;

  for (int region = begin; region < end; ++region) {
    uint32_t const key = ordered_score(score_row[region], region);
    if (key > threshold_key) {
      staged_regions[staged_count++] = region;
    } else if (key == threshold_key) {
      ++local_equal;
    }
    // Always call __any_sync with the full warp so the SIMT mask stays
    // convergent even when only a subset of lanes found a greater region.
    if (__any_sync(0xFFFFFFFF, staged_count == kThreadStaging)) {
      int warp_exclusive = 0;
      int const base_slot =
          claim_warp_output_slots(staged_count, thread, storage,
                                  warp_exclusive);
#pragma unroll
      for (int i = 0; i < kThreadStaging; ++i) {
        if (i < staged_count) {
          int const slot = base_slot + warp_exclusive + i;
          if (slot < storage.selected_base) {
            output_row[slot] = staged_regions[i];
          }
        }
      }
      staged_count = 0;
    }
  }
  {
    int warp_exclusive = 0;
    int const base_slot =
        claim_warp_output_slots(staged_count, thread, storage, warp_exclusive);
#pragma unroll
    for (int i = 0; i < kThreadStaging; ++i) {
      if (i < staged_count) {
        int const slot = base_slot + warp_exclusive + i;
        if (slot < storage.selected_base) {
          output_row[slot] = staged_regions[i];
        }
      }
    }
  }
  int const higher_equal =
      equal_count_higher_equal(local_equal, thread, storage);
  int const local_take =
      min(local_equal, max(storage.remain - higher_equal, 0));
  int emitted = 0;
  for (int region = end - 1; region >= begin && emitted < local_take;
       --region) {
    if (ordered_score(score_row[region], region) == threshold_key) {
      int const slot = storage.selected_base + higher_equal + emitted;
      if (slot < topk) {
        output_row[slot] = region;
      }
      ++emitted;
    }
  }
}

// Merge kernel: 4 rounds [11 score, 11 score, 10 score, 10 region].
__global__ void dsa_radix_merge_kernel(
    float const* scores, int64_t const* query_positions,
    int const* candidate_input, int candidate_stride,
    int const* block_table, int* region_output, int* start_output, int heads,
    int regions, int topk, int region_block, int page_size,
    int block_table_stride, int block_table_length, int num_chunks,
    bool packed) {
  extern __shared__ char raw_shared[];
  auto& storage = *reinterpret_cast<DsaMergeRadixShared*>(raw_shared);
  int const work = blockIdx.x;
  int const request = work / heads;
  int const head = work - request * heads;
  int const thread = threadIdx.x;
  int const output_width = topk + 1;
  int const current = min(
      max(int(query_positions[request]), 0) / region_block, regions - 1);
  int const target = min(topk, current);
  int const candidate_count = num_chunks * topk;
  float const* score_row = scores + work * regions;
  int const* candidate_row = candidate_input + work * candidate_stride;

  for (int slot = thread; slot < output_width; slot += kThreads) {
    int const offset = work * output_width + slot;
    region_output[offset] = -1;
    start_output[offset] = -1;
  }
  if (thread == 0) {
    storage.score_prefix_value = 0;
    storage.score_prefix_mask = 0;
    storage.region_prefix_value = 0;
    storage.region_prefix_mask = 0;
    storage.remain = target;
    storage.selected_base = 0;
  }
  __syncthreads();

  // Rounds 0-2: score bits [11, 11, 10]; Round 3: region bits [10].
  constexpr int kMergeRounds = 4;
#pragma unroll
  for (int round = 0; round < kMergeRounds; ++round) {
    bool const score_round = round < 3;
    int n_bins, bits, shift;
    if (round == 0) {
      n_bins = kCoarseRadix; bits = 11; shift = 21;  // score [31:21]
    } else if (round == 1) {
      n_bins = kCoarseRadix; bits = 11; shift = 10;  // score [20:10]
    } else if (round == 2) {
      n_bins = kRadix; bits = 10; shift = 0;         // score [9:0]
    } else {
      n_bins = kRadix; bits = 10; shift = 22;        // region [31:22]
    }
    uint32_t const bin_mask = (1u << bits) - 1u;

    for (int bin = thread; bin < n_bins; bin += kThreads) {
      storage.histogram[bin] = 0;
    }
    __syncthreads();

    uint32_t const score_prefix_value = storage.score_prefix_value;
    uint32_t const score_prefix_mask = storage.score_prefix_mask;
    uint32_t const region_prefix_value = storage.region_prefix_value;
    uint32_t const region_prefix_mask = storage.region_prefix_mask;
    for (int index = thread; index < candidate_count; index += kThreads) {
      int const region = candidate_row[index];
      if (region < 0 || region >= current) continue;
      uint32_t const score_key = ordered_score(score_row[region], region);
      uint32_t const region_key = static_cast<uint32_t>(region);
      if ((score_key & score_prefix_mask) == score_prefix_value &&
          (region_key & region_prefix_mask) == region_prefix_value) {
        uint32_t const key = score_round ? score_key : region_key;
        atomicAdd(&storage.histogram[int((key >> shift) & bin_mask)], 1);
      }
    }
    __syncthreads();

    if (n_bins == kCoarseRadix) {
      choose_threshold_parallel<kCoarseRadix>(storage, thread);
    } else {
      choose_threshold_parallel<kRadix>(storage, thread);
    }

    if (thread == 0 && storage.remain > 0) {
      uint32_t const round_mask = bin_mask << shift;
      if (score_round) {
        storage.score_prefix_mask |= round_mask;
        storage.score_prefix_value =
            (storage.score_prefix_value & ~round_mask) |
            (uint32_t(storage.threshold) << shift);
      } else {
        storage.region_prefix_mask |= round_mask;
        storage.region_prefix_value =
            (storage.region_prefix_value & ~round_mask) |
            (uint32_t(storage.threshold) << shift);
      }
      storage.selected_base += storage.count_greater;
      storage.remain -= storage.count_greater;
    }
    __syncthreads();
  }

  uint32_t const threshold_score = storage.score_prefix_value;
  uint32_t const threshold_region = storage.region_prefix_value;
  if (thread == 0) {
    storage.output_counter = 0;
  }
  __syncthreads();
  for (int index = thread; index < candidate_count; index += kThreads) {
    int const region = candidate_row[index];
    if (region < 0 || region >= current) continue;
    uint32_t const score_key = ordered_score(score_row[region], region);
    bool const selected =
        score_key > threshold_score ||
        (score_key == threshold_score &&
         static_cast<uint32_t>(region) >= threshold_region);
    if (selected) {
      int const slot = atomicAdd(&storage.output_counter, 1);
      if (slot < target) {
        store_region(region, slot, request, head, heads, output_width,
                     block_table, block_table_stride, block_table_length,
                     page_size / region_block, region_block, packed,
                     region_output, start_output);
      }
    }
  }

  if (thread == 0) {
    store_region(current, topk, request, head, heads, output_width,
                 block_table, block_table_stride, block_table_length,
                 page_size / region_block, region_block, packed,
                 region_output, start_output);
  }
}

/// \brief Host launcher of one thread-count specialization. The JIT entry
/// \ref sglang::step5_dsa::radix_select dispatches between the two instances
/// instantiated by the double include.
void dsa_radix_select(
    const tvm::ffi::TensorView scores,
    const tvm::ffi::TensorView query_positions,
    const tvm::ffi::TensorView block_table,
    const tvm::ffi::TensorView candidate_workspace,
    const tvm::ffi::TensorView region_output,
    const tvm::ffi::TensorView start_output, int64_t region_block,
    int64_t page_size, int64_t topk, bool packed) {
  using namespace host;

  auto S_query = SymbolicSize{"query"};
  auto S_heads = SymbolicSize{"heads"};
  auto S_regions = SymbolicSize{"regions"};
  // SizeRef only binds lvalues, so the shapes reused across matchers are named.
  auto S_pages = SymbolicSize{"pages"};
  auto S_capacity = SymbolicSize{"capacity"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({S_query, S_heads, S_regions})
      .with_dtype<float>()
      .with_device(device)
      .verify(scores);
  TensorMatcher({S_query})
      .with_dtype<int64_t>()
      .with_device(device)
      .verify(query_positions);
  if (block_table.dim() == 1) {
    TensorMatcher({S_pages})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(block_table);
  } else {
    TensorMatcher({S_query, S_pages})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(block_table);
  }
  TensorMatcher({S_query, S_heads, S_capacity})
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(candidate_workspace);
  TensorMatcher({S_query, S_heads, topk + 1})
      .with_dtype<int32_t>()
      .with_device(device)
      .verify(region_output)
      .verify(start_output);
  CHECK_HOST(region_block > 0 && page_size >= region_block &&
             page_size % region_block == 0)
      << "invalid Step5 DSA region/page geometry";
  CHECK_HOST(topk > 0 && topk <= 1024 && S_regions.unwrap() > 0 &&
             S_regions.unwrap() <= std::numeric_limits<int>::max())
      << "invalid Step5 DSA radix top-k geometry";

  int const query_rows = static_cast<int>(S_query.unwrap());
  int const heads = static_cast<int>(S_heads.unwrap());
  int const regions = static_cast<int>(S_regions.unwrap());
  int const candidate_stride = static_cast<int>(candidate_workspace.size(2));
  // A rank-2 block table holds one row of pages per query token (MTP draft
  // batching); rank-1 shares a single row across the batch, hence stride 0.
  int const block_table_stride =
      block_table.dim() == 2 ? static_cast<int>(block_table.stride(0)) : 0;
  int const block_table_length =
      static_cast<int>(block_table.size(block_table.dim() - 1));
  hggcStream_t stream = LaunchKernel::resolve_device(scores.device());

  // Request extra shared memory for the 4096-bin histogram.
  constexpr size_t kSelectorSmem = sizeof(DsaRadixShared);
  constexpr size_t kMergeSmem = sizeof(DsaMergeRadixShared);
  static_assert(kSelectorSmem <= 48 * 1024,
                "selector shared memory exceeds default 48KB limit");
  static_assert(kMergeSmem <= 48 * 1024,
                "merge shared memory exceeds default 48KB limit");
  // Both structs fit within the default 48KB dynamic shared memory limit,
  // so no cudaFuncSetAttribute is needed.

  if (regions <= kStreamingChunkRegions) {
    dsa_radix_selector_kernel<<<query_rows * heads, kThreads, kSelectorSmem,
                                stream>>>(
        static_cast<const float*>(scores.data_ptr()),
        static_cast<const int64_t*>(query_positions.data_ptr()),
        static_cast<const int*>(block_table.data_ptr()),
        static_cast<int*>(region_output.data_ptr()),
        static_cast<int*>(start_output.data_ptr()), heads, regions,
        static_cast<int>(topk), static_cast<int>(region_block),
        static_cast<int>(page_size), block_table_stride, block_table_length,
        packed);
  } else {
    int const num_chunks =
        (regions + kStreamingChunkRegions - 1) / kStreamingChunkRegions;
    CHECK_HOST(int64_t(num_chunks) * topk <= candidate_stride)
        << "candidate workspace is too small for Step5 DSA multi-CTA radix: "
        << "need " << int64_t(num_chunks) * topk << ", got "
        << candidate_stride;
    dsa_radix_chunk_kernel<<<query_rows * heads * num_chunks, kThreads,
                             kSelectorSmem, stream>>>(
        static_cast<const float*>(scores.data_ptr()),
        static_cast<const int64_t*>(query_positions.data_ptr()),
        static_cast<int*>(candidate_workspace.data_ptr()), heads, regions,
        static_cast<int>(topk), static_cast<int>(region_block), num_chunks,
        candidate_stride);
    dsa_radix_merge_kernel<<<query_rows * heads, kThreads, kMergeSmem,
                             stream>>>(
        static_cast<const float*>(scores.data_ptr()),
        static_cast<const int64_t*>(query_positions.data_ptr()),
        static_cast<int*>(candidate_workspace.data_ptr()), candidate_stride,
        static_cast<const int*>(block_table.data_ptr()),
        static_cast<int*>(region_output.data_ptr()),
        static_cast<int*>(start_output.data_ptr()), heads, regions,
        static_cast<int>(topk), static_cast<int>(region_block),
        static_cast<int>(page_size), block_table_stride, block_table_length,
        num_chunks, packed);
  }
  CHECK_CUDA(hggcGetLastError()) << "Step5 DSA radix launch failed";
}

}  // namespace dsa_selector_w256 / dsa_selector_w1024
}  // namespace sglang
