// SPDX-License-Identifier: Apache-2.0
//
// JIT entry for the Step5 DSA radix region selector. Both thread-count
// specializations (256 and 1024 threads) are instantiated in this single
// translation unit by including the implementation header twice; radix_select
// dispatches between them from the workload shape.

#pragma once

#define DSA_SELECTOR_THREADS 1024
#include "dsa_radix_selector_impl.cuh"
#undef DSA_SELECTOR_THREADS

#define DSA_SELECTOR_THREADS 256
#include "dsa_radix_selector_impl.cuh"
#undef DSA_SELECTOR_THREADS

#include <cstdlib>
#include <cstring>

namespace sglang {
namespace step5_dsa {

// TensorMatcher / SymbolicSize / SymbolicDevice / LaunchKernel live in host::.
using namespace host;
namespace {

// Prefill-style batches saturate 256 threads; decode and very large contexts
// keep 1024 so one CTA still covers a full radix round.
bool use_w256(const tvm::ffi::TensorView scores) {
  static const int64_t forced = []() -> int64_t {
    const char* env = std::getenv("SGLANG_STEP5_DSA_SELECTOR_THREADS");
    if (env == nullptr) return 0;
    if (std::strcmp(env, "256") == 0) return 256;
    if (std::strcmp(env, "1024") == 0) return 1024;
    return 0;
  }();
  if (forced != 0) return forced == 256;
  int64_t const work_items = scores.size(0) * scores.size(1);
  return work_items >= 1000 && scores.size(2) <= 16384;
}

}  // namespace

/// \brief Select the top-k logical regions of a Step5 DSA score row.
///
/// Ranks only the live prefix [0, position / region_block) of each row, writes
/// the selected history regions to slots [0, k) and the current region to slot
/// `topk`; every unwritten slot stays -1. Slots inside [0, k) are unordered:
/// the staging pass hands them out with atomics, so only the selected set is
/// stable across runs. Ties fold the region id into the score as
/// `score + id * 1e-9`, so they resolve toward the larger id while the score
/// magnitude leaves room for that step in FP32. Both thread-count
/// specializations are instantiated above; this entry picks one from the
/// workload shape (SGLANG_STEP5_DSA_SELECTOR_THREADS overrides it).
///
/// \param scores contiguous FP32 [query, heads, regions] scores, -0.0 already
///     folded to +0.0
/// \param query_positions contiguous int64 [query] logical token positions
/// \param block_table contiguous int32 [pages] or [query, pages]
///     logical-to-physical pages; only read when \p packed
/// \param candidate_workspace contiguous int32 [query, heads, capacity] radix
///     scratch, used when \p regions exceeds 16384
/// \param region_output contiguous int32 [query, heads, topk + 1] selected
///     regions; -1 padding
/// \param start_output contiguous int32 [query, heads, topk + 1] logical token
///     starts; -1 padding
/// \param region_block tokens per logical region; page_size must be a positive
///     multiple of it
/// \param page_size tokens per physical page
/// \param topk history regions to keep, at most 1024
/// \param packed translate region ids through \p block_table into physical
///     region ids
void radix_select(const tvm::ffi::TensorView scores,
                  const tvm::ffi::TensorView query_positions,
                  const tvm::ffi::TensorView block_table,
                  const tvm::ffi::TensorView candidate_workspace,
                  const tvm::ffi::TensorView region_output,
                  const tvm::ffi::TensorView start_output,
                  int64_t region_block, int64_t page_size, int64_t topk,
                  bool packed) {
  if (use_w256(scores)) {
    dsa_selector_w256::dsa_radix_select(
        scores, query_positions, block_table, candidate_workspace,
        region_output, start_output, region_block, page_size, topk, packed);
  } else {
    dsa_selector_w1024::dsa_radix_select(
        scores, query_positions, block_table, candidate_workspace,
        region_output, start_output, region_block, page_size, topk, packed);
  }
}

}  // namespace step5_dsa
}  // namespace sglang
