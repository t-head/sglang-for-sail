// Copyright 2023-2026 SGLang Team
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Push a layer's local expert slices to peer-visible staging on a copy stream.
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector>

namespace {
void enqueue_memcpy_batch(
    const std::vector<int64_t>& src,
    const std::vector<int64_t>& dst,
    const std::vector<int64_t>& sizes,
    int64_t stream_handle) {
  TORCH_CHECK(stream_handle && src.size() == dst.size() && src.size() == sizes.size(), "invalid EPLB copy batch");
  for (size_t i = 0; i < sizes.size(); ++i) {
    TORCH_CHECK(src[i] && dst[i] && src[i] != dst[i] && sizes[i] > 0, "invalid expert slice at index ", i);
  }
  auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
  for (size_t i = 0; i < sizes.size(); ++i) {
    auto err = cudaMemcpyAsync(
        reinterpret_cast<void*>(dst[i]), reinterpret_cast<const void*>(src[i]), sizes[i], cudaMemcpyDefault, stream);
    TORCH_CHECK(err == cudaSuccess, "EPLB cudaMemcpyAsync: ", cudaGetErrorString(err));
  }
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("enqueue_memcpy_batch", &enqueue_memcpy_batch, pybind11::call_guard<pybind11::gil_scoped_release>());
}
