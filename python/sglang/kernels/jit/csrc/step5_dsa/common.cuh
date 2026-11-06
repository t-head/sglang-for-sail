// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

// The tc02 ldmatrix and the 2d.tile AIU bulk copy exist only on the ppu0015
// core, which reports capability 8.9. The PPU compiler emits every translation
// unit for both ppu0010 and ppu0015, so the attention device code that spells
// those instructions is guarded: the ppu0010 pass gets inert stand-ins and the
// launchers refuse that core at runtime instead of computing a wrong answer.
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ >= 150
  #define STEP5_DSA_TC02 1
#else
  #define STEP5_DSA_TC02 0
#endif

namespace sglang {
namespace step5_dsa {

// TensorMatcher / SymbolicSize / SymbolicDevice / LaunchKernel live in host::.
using namespace host;

inline void check_int32(std::initializer_list<int64_t> values) {
  for (const int64_t value : values) {
    if (value < 0 || value > std::numeric_limits<int>::max()) {
      host::Panic("DSA geometry exceeds int32 indexing capacity");
    }
  }
}

inline void check_scale(double scale) {
  if (!(std::isfinite(scale) && static_cast<float>(scale) > 0 &&
        scale <= std::numeric_limits<float>::max())) {
    host::Panic("scaling must be finite and positive in float32");
  }
}

// SGLang stores its selection as packed int64 words: the low word holds the
// logical start and the high word the physical region. Split them once per
// layer into the two contiguous int32 planes the attention kernels index, so
// the kernels never read the packed layout themselves. Both the paged decode
// and the prefill modules export this bridge as ``unpack_regions``.
namespace {

__global__ void unpack_region_words_kernel(const int64_t* __restrict__ packed,
                                           int32_t* __restrict__ physical,
                                           int32_t* __restrict__ starts,
                                           int64_t words) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= words) return;
  const int64_t word = packed[index];
  physical[index] = static_cast<int32_t>(word >> 32);
  starts[index] = static_cast<int32_t>(word);
}

}  // namespace

/// \brief Split packed region words into the int32 physical-region and
///     logical-start planes consumed by the paged decode and prefill attention
///     kernels.
///
/// The callers disagree on rank: paged decode hands over ``[rows, regions]``
/// while prefill hands over ``[rows, 1]``. Both carry the same word sequence,
/// so the bridge validates element counts and the shared device instead of a
/// fixed shape.
void unpack_packed_regions(const tvm::ffi::TensorView packed,
                           const tvm::ffi::TensorView physical_regions,
                           const tvm::ffi::TensorView logical_starts) {
  CHECK_HOST(is_type<int64_t>(packed.dtype()))
      << "packed Step5 selection must be int64";
  CHECK_HOST(is_type<int32_t>(physical_regions.dtype()) &&
             is_type<int32_t>(logical_starts.dtype()))
      << "Step5 region planes must be int32";
  const int64_t words = packed.numel();
  CHECK_HOST(physical_regions.numel() == words &&
             logical_starts.numel() == words)
      << "Step5 region planes must hold " << words << " words, got "
      << physical_regions.numel() << " and " << logical_starts.numel();
  const DLDevice device = packed.device();
  CHECK_HOST(device.device_type == kDLCUDA &&
             physical_regions.device().device_type == kDLCUDA &&
             logical_starts.device().device_type == kDLCUDA &&
             physical_regions.device().device_id == device.device_id &&
             logical_starts.device().device_id == device.device_id)
      << "Step5 selection and its planes must share one CUDA device";
  if (words == 0) return;
  hggcStream_t stream = LaunchKernel::resolve_device(device);
  constexpr int kThreads = 256;
  unpack_region_words_kernel<<<(words + kThreads - 1) / kThreads, kThreads, 0,
                               stream>>>(
      static_cast<const int64_t*>(packed.data_ptr()),
      static_cast<int32_t*>(physical_regions.data_ptr()),
      static_cast<int32_t*>(logical_starts.data_ptr()), words);
  CHECK_CUDA(hggcGetLastError()) << "Step5 DSA unpack regions launch failed";
}

}  // namespace step5_dsa
}  // namespace sglang
