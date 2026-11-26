// Derived producer/layout code from FlashKDA, MIT License.
// Copyright (c) 2026 MoonshotAI
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// SPDX-License-Identifier: Apache-2.0 AND MIT
// Read-only verify records FP32 corrections, normalized keys and decay.
// Accepted-prefix recovery never recomputes the delta rule.
#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cassert>
#include <cstdint>

namespace sglang {
namespace replay_ppu {


SGL_DEVICE float sum_xor(float x) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1)
    x += __shfl_xor_sync(0xffffffffu, x, d);
  return x;
}
SGL_DEVICE float sum_down(float x) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1)
    x += __shfl_down_sync(0xffffffffu, x, d);
  return x;
}
SGL_DEVICE float sigmoid(float x) {
#ifdef __HGGC_ARCH__
  // Same intrinsic used by the validated local FlashKDA PPU verify.
  return __ppu_sgmdf(x);
#else
  return 1.f / (1.f + __expf(-x));
#endif
}

template <typename State>
struct Verify {
  const bf16_t *q, *k, *v, *a, *b;
  const float *A, *dt;
  const State* state;
  State* snapshots;
  const int32_t* indices;
  float *correction, *key, *decay;
  bf16_t* out;
  int32_t H, T, L, slots;
  int64_t qs, ks, vs, as;
  float lower;
};

using BF16 = bf16_t;
constexpr int kD = 128;
SGL_DEVICE float half_sum_down(float x) {
#pragma unroll
  for (int d = 8; d > 0; d >>= 1)
    x += __shfl_down_sync(0xffffffffu, x, d, 16);
  return x;
}
SGL_DEVICE float half_sum_all(float x) {
#pragma unroll
  for (int d = 8; d > 0; d >>= 1)
    x += __shfl_xor_sync(0xffffffffu, x, d, 16);
  return x;
}
SGL_DEVICE float2 bf16x2_to_float2(uint32_t raw) {
  return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&raw));
}

SGL_DEVICE float4 load_bf16x4(BF16 const* ptr) {
  uint2 raw = *reinterpret_cast<uint2 const*>(ptr);
  float2 lo = bf16x2_to_float2(raw.x);
  float2 hi = bf16x2_to_float2(raw.y);
  return make_float4(lo.x, lo.y, hi.x, hi.y);
}

template <typename StateT>
SGL_DEVICE float4 load_state4(StateT const* ptr) {
  if constexpr (sizeof(StateT) == sizeof(float)) {
    return *reinterpret_cast<float4 const*>(ptr);
  } else {
    return load_bf16x4(reinterpret_cast<BF16 const*>(ptr));
  }
}

template <int Rows>
SGL_DEVICE void produce_token(
    BF16 const* q,
    BF16 const* k,
    BF16 const* v,
    BF16 const* gate,
    BF16 const* beta,
    float const* dt_bias,
    float exp_A,
    float scale,
    float lower_bound,
    int token,
    int head,
    int H,
    int64_t q_token_stride,
    int64_t k_token_stride,
    int64_t v_token_stride,
    int64_t gate_token_stride,
    int row_base,
    int warp,
    int lane,
    float* sm_q,
    float* sm_k,
    float* sm_decay,
    float* sm_beta,
    BF16* sm_v) {
  // The caller passes token as the already packed B*T token index.
  int64_t const q_base = int64_t(token) * q_token_stride + head * kD;
  int64_t const k_base = int64_t(token) * k_token_stride + head * kD;
  int64_t const v_base = int64_t(token) * v_token_stride + head * kD;
  int64_t const gate_base = int64_t(token) * gate_token_stride + head * kD;
  if (warp == 0) {
    // Run the independent Q and K reductions in the two 16-lane groups.
    // This frees a second warp for the SFU-heavy gate/decay transform.
    int const half = lane >> 4;
    int const half_lane = lane & 15;
    int const col = half_lane * 8;
    BF16 const* src = half == 0 ? q + q_base : k + k_base;
    float* dst = half == 0 ? sm_q : sm_k;
    float4 x0 = load_bf16x4(src + col);
    float4 x1 = load_bf16x4(src + col + 4);
    float sq =
        x0.x * x0.x + x0.y * x0.y + x0.z * x0.z + x0.w * x0.w + x1.x * x1.x + x1.y * x1.y + x1.z * x1.z + x1.w * x1.w;
    float inv = rsqrtf(half_sum_all(sq) + 1.0e-6f);
    inv *= half == 0 ? scale : 1.0f;
    x0.x *= inv;
    x0.y *= inv;
    x0.z *= inv;
    x0.w *= inv;
    x1.x *= inv;
    x1.y *= inv;
    x1.z *= inv;
    x1.w *= inv;
    *reinterpret_cast<float4*>(dst + col) = x0;
    *reinterpret_cast<float4*>(dst + col + 4) = x1;
  } else if (warp == 1 || warp == 2) {
    int const col = (warp - 1) * 64 + lane * 2;
    float2 a = bf16x2_to_float2(*reinterpret_cast<uint32_t const*>(gate + gate_base + col));
    float2 dt = *reinterpret_cast<float2 const*>(dt_bias + head * kD + col);
    float2 d = make_float2(
        __expf(lower_bound * sigmoid(exp_A * (a.x + dt.x))),
        __expf(lower_bound * sigmoid(exp_A * (a.y + dt.y))));
    *reinterpret_cast<float2*>(sm_decay + col) = d;
    if (warp == 1 && lane == 0) {
      sm_beta[0] = sigmoid(float(beta[token * H + head]));
    }
  } else if (warp == 3) {
    // The scratch tail is 16-byte aligned without increasing its rounded
    // allocation, so copy eight BF16 V values per active lane.
    if (lane < Rows / 8) {
      int const row = lane * 8;
      *reinterpret_cast<uint4*>(sm_v + row) = *reinterpret_cast<uint4 const*>(v + v_base + row_base + row);
    }
  }
}

template <typename State>
SGL_DEVICE void save4(State* dst, float4 h) {
  if constexpr (sizeof(State) == 4)
    *reinterpret_cast<float4*>(dst) = h;
  else {
    const auto lo = __floats2bfloat162_rn(h.x, h.y), hi = __floats2bfloat162_rn(h.z, h.w);
    *reinterpret_cast<uint2*>(dst) =
        make_uint2(*reinterpret_cast<const uint32_t*>(&lo), *reinterpret_cast<const uint32_t*>(&hi));
  }
}

SGL_DEVICE float dot4(float4 h, float4 k) {
  return h.x * k.x + h.y * k.y + h.z * k.z + h.w * k.w;
}

template <typename State>
__global__ void verify_kernel(Verify<State> p) {
  constexpr int Rows = 16;
  const int row_base = blockIdx.x * Rows, head = blockIdx.y, batch = blockIdx.z;
  const int lane = threadIdx.x, warp = threadIdx.y, half = lane >> 4, hl = lane & 15, col = hl * 8;
  const int slot = p.indices[batch];
  if (slot < 0) {
    for (int t = 0; t < p.T; ++t)
      if (warp == 0 && lane < Rows)
        p.out[((int64_t(batch) * p.T + t) * p.H + head) * 128 + row_base + lane] = BF16(0.f);
    return;
  }
  assert(slot < p.slots);
  __shared__ float sq[128], sk[128], sd[128], sb[4];
  __shared__ BF16 sv[Rows];
  float4 h0[2], h1[2];
#pragma unroll
  for (int g = 0; g < 2; ++g) {
    const int row = row_base + warp * 2 + half + g * 8;
    const State* src = p.state + (int64_t(slot) * p.H + head) * 128 * 128 + row * 128 + col;
    h0[g] = load_state4(src);
    h1[g] = load_state4(src + 4);
  }
  const float exp_A = __shfl_sync(0xffffffffu, lane == 0 ? __expf(p.A[head]) : 0.f, 0);
  for (int t = 0; t < p.T; ++t) {
    const int64_t ring = ((int64_t(slot) * p.H + head) * p.L + t) * 128;
    produce_token<Rows>(
        p.q,
        p.k,
        p.v,
        p.a,
        p.b,
        p.dt,
        exp_A,
        0.08838834764831844f,
        p.lower,
        batch * p.T + t,
        head,
        p.H,
        p.qs,
        p.ks,
        p.vs,
        p.as,
        row_base,
        warp,
        lane,
        sq,
        sk,
        sd,
        sb,
        sv);
    __syncthreads();
    if (blockIdx.x == 0 && warp == 3) {
      *reinterpret_cast<float4*>(p.key + ring + lane * 4) = *reinterpret_cast<const float4*>(sk + lane * 4);
      *reinterpret_cast<float4*>(p.decay + ring + lane * 4) = *reinterpret_cast<const float4*>(sd + lane * 4);
    }
    const float4 q0 = *reinterpret_cast<const float4*>(sq + col), q1 = *reinterpret_cast<const float4*>(sq + col + 4);
    const float4 k0 = *reinterpret_cast<const float4*>(sk + col), k1 = *reinterpret_cast<const float4*>(sk + col + 4);
    const float4 d0 = *reinterpret_cast<const float4*>(sd + col), d1 = *reinterpret_cast<const float4*>(sd + col + 4);
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      const int row = warp * 2 + half + g * 8;
      float4& x0 = h0[g];
      float4& x1 = h1[g];
      x0.x = __fmul_rn(x0.x, d0.x);
      x0.y = __fmul_rn(x0.y, d0.y);
      x0.z = __fmul_rn(x0.z, d0.z);
      x0.w = __fmul_rn(x0.w, d0.w);
      x1.x = __fmul_rn(x1.x, d1.x);
      x1.y = __fmul_rn(x1.y, d1.y);
      x1.z = __fmul_rn(x1.z, d1.z);
      x1.w = __fmul_rn(x1.w, d1.w);
      float hk = half_sum_down(dot4(x0, k0) + dot4(x1, k1));
      hk = __shfl_sync(0xffffffffu, hk, half * 16);
      const float u = (float(sv[row]) - hk) * sb[0];
      if (hl == 0) p.correction[ring + row_base + row] = u;
      x0.x = __fmaf_rn(k0.x, u, x0.x);
      x0.y = __fmaf_rn(k0.y, u, x0.y);
      x0.z = __fmaf_rn(k0.z, u, x0.z);
      x0.w = __fmaf_rn(k0.w, u, x0.w);
      x1.x = __fmaf_rn(k1.x, u, x1.x);
      x1.y = __fmaf_rn(k1.y, u, x1.y);
      x1.z = __fmaf_rn(k1.z, u, x1.z);
      x1.w = __fmaf_rn(k1.w, u, x1.w);
      const float hq = half_sum_down(dot4(x0, q0) + dot4(x1, q1));
      if (hl == 0) p.out[((int64_t(batch) * p.T + t) * p.H + head) * 128 + row_base + row] = BF16(hq);
      if (p.snapshots) {
        State* dst = p.snapshots + ((int64_t(batch) * p.T + t) * p.H + head) * 128 * 128 + (row_base + row) * 128 + col;
        save4(dst, x0);
        save4(dst + 4, x1);
      }
    }
    __syncthreads();
  }
}

template <typename State>
__global__ void commit_kernel(
    State* state,
    const float* correction,
    const float* key,
    const float* decay,
    const int32_t* indices,
    const int32_t* accepts,
    int H,
    int L,
    int slots,
    int64_t state_layer,
    int64_t ring_layer,
    const int32_t* track_indices,
    const int32_t* track_steps) {
  const int layer = blockIdx.z / H, head = blockIdx.z % H, batch = blockIdx.y;
  const int lane = threadIdx.x, warp = threadIdx.y, col = lane * 4;
  const int slot = indices[batch], n = accepts[batch];
  if (slot < 0 || n <= 0) return;
  assert(slot < slots && n <= L);
  const int track = track_indices ? track_indices[batch] : -1;
  const int track_step = track_steps ? track_steps[batch] : -1;
  assert(track < slots);
  const int64_t base = int64_t(layer) * state_layer + (int64_t(slot) * H + head) * 128 * 128;
  const int64_t ringbase = int64_t(layer) * ring_layer + (int64_t(slot) * H + head) * L * 128;
  float4 h[8];
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    const int row = blockIdx.x * 32 + warp + r * 4;
    const State* src = state + base + row * 128 + col;
    if constexpr (sizeof(State) == 4)
      h[r] = *reinterpret_cast<const float4*>(src);
    else {
      const uint2 raw = *reinterpret_cast<const uint2*>(src);
      const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
      const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
      h[r] = make_float4(lo.x, lo.y, hi.x, hi.y);
    }
  }
  for (int t = 0; t < n; ++t) {
    const float4 d = *reinterpret_cast<const float4*>(decay + ringbase + int64_t(t) * 128 + col);
    const float4 k = *reinterpret_cast<const float4*>(key + ringbase + int64_t(t) * 128 + col);
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const int row = blockIdx.x * 32 + warp + r * 4;
      const float u = correction[ringbase + int64_t(t) * 128 + row];
      h[r].x = __fmaf_rn(k.x, u, __fmul_rn(h[r].x, d.x));
      h[r].y = __fmaf_rn(k.y, u, __fmul_rn(h[r].y, d.y));
      h[r].z = __fmaf_rn(k.z, u, __fmul_rn(h[r].z, d.z));
      h[r].w = __fmaf_rn(k.w, u, __fmul_rn(h[r].w, d.w));
      if (track >= 0 && t == track_step) {
        State* dst = state + int64_t(layer) * state_layer + (int64_t(track) * H + head) * 128 * 128 + row * 128 + col;
        save4(dst, h[r]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    const int row = blockIdx.x * 32 + warp + r * 4;
    State* dst = state + base + row * 128 + col;
    if constexpr (sizeof(State) == 4)
      *reinterpret_cast<float4*>(dst) = h[r];
    else {
      const auto lo = __floats2bfloat162_rn(h[r].x, h[r].y);
      const auto hi = __floats2bfloat162_rn(h[r].z, h[r].w);
      *reinterpret_cast<uint2*>(dst) =
          make_uint2(*reinterpret_cast<const uint32_t*>(&lo), *reinterpret_cast<const uint32_t*>(&hi));
    }
  }
}
}  // namespace replay_ppu

/// Read-only dense-chain KDA verify with lossless correction records.
void replay_verify_impl(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView dt,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView correction,
    tvm::ffi::TensorView key,
    tvm::ffi::TensorView decay,
    tvm::ffi::TensorView out,
    double lower,
    const tvm::ffi::TensorView* snapshots) {
  using namespace host;
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  auto B = SymbolicSize{"batch"}, N = SymbolicSize{"tokens"}, H = SymbolicSize{"heads"};
  auto S = SymbolicSize{"slots"}, L = SymbolicSize{"ring_len"};
  auto q0 = SymbolicSize{"q0"}, q1 = SymbolicSize{"q1"};
  auto k0 = SymbolicSize{"k0"}, k1 = SymbolicSize{"k1"};
  auto v0 = SymbolicSize{"v0"}, v1 = SymbolicSize{"v1"};
  auto a0 = SymbolicSize{"a0"}, a1 = SymbolicSize{"a1"};
  TensorMatcher({1, N, H, 128}).with_dtype<bf16_t>().with_device(device).with_strides({q0, q1, 128, 1}).verify(q);
  TensorMatcher({1, N, H, 128}).with_dtype<bf16_t>().with_device(device).with_strides({k0, k1, 128, 1}).verify(k);
  TensorMatcher({1, N, H, 128}).with_dtype<bf16_t>().with_device(device).with_strides({v0, v1, 128, 1}).verify(v);
  TensorMatcher({1, N, H, 128}).with_dtype<bf16_t>().with_device(device).with_strides({a0, a1, 128, 1}).verify(a);
  TensorMatcher({1, N, H}).with_dtype<bf16_t>().with_device(device).verify(b);
  TensorMatcher({H}).with_dtype<float>().with_device(device).verify(A);
  TensorMatcher({H, 128}).with_dtype<float>().with_device(device).verify(dt);
  TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(indices);
  TensorMatcher({S, H, 128, 128}).with_dtype<bf16_t, float>().with_device(device).verify(state);
  TensorMatcher({S, H, L, 128}).with_dtype<float>().with_device(device).verify(correction).verify(key).verify(decay);
  TensorMatcher({1, N, H, 128}).with_dtype<bf16_t>().with_device(device).verify(out);
  CHECK_HOST(B.unwrap() == 0 || N.unwrap() % B.unwrap() == 0) << "dense chains required";
  if (B.unwrap() == 0) return;
  for (const auto tensor : {q, k, a}) {
    CHECK_HOST(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 8 == 0 && tensor.stride(1) % 4 == 0)
        << "KDA activation views must be 8-byte aligned";
  }
  CHECK_HOST(reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0 && v.stride(1) % 8 == 0)
      << "KDA V views must be 16-byte aligned";
  CHECK_HOST(reinterpret_cast<uintptr_t>(dt.data_ptr()) % 8 == 0) << "KDA dt bias must be aligned";
  const auto state_alignment = is_type<bf16_t>(state.dtype()) ? 8 : 16;
  CHECK_HOST(reinterpret_cast<uintptr_t>(state.data_ptr()) % state_alignment == 0)
      << "KDA state must be vector aligned";
  for (const auto tensor : {correction, key, decay}) {
    CHECK_HOST(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0)
        << "KDA compact records must be 16-byte aligned";
  }
  const int T = N.unwrap() / B.unwrap();
  if (snapshots && snapshots->numel() > 0) {
    TensorMatcher({B, T, H, 128, 128}).with_dtype<bf16_t, float>().with_device(device).verify(*snapshots);
    CHECK_HOST(snapshots->dtype() == state.dtype()) << "snapshot/state dtype mismatch";
    CHECK_HOST(reinterpret_cast<uintptr_t>(snapshots->data_ptr()) % state_alignment == 0)
        << "KDA snapshots must be aligned";
  }
  CHECK_HOST(T > 0 && T <= L.unwrap() && T <= 16) << "unsupported chain length";
#define VERIFY_TYPE(TY)                                                                                            \
  do {                                                                                                             \
    auto p = replay_ppu::Verify<TY>{                                                                               \
        static_cast<const bf16_t*>(q.data_ptr()),                                                                  \
        static_cast<const bf16_t*>(k.data_ptr()),                                                                  \
        static_cast<const bf16_t*>(v.data_ptr()),                                                                  \
        static_cast<const bf16_t*>(a.data_ptr()),                                                                  \
        static_cast<const bf16_t*>(b.data_ptr()),                                                                  \
        static_cast<const float*>(A.data_ptr()),                                                                   \
        static_cast<const float*>(dt.data_ptr()),                                                                  \
        static_cast<const TY*>(state.data_ptr()),                                                                  \
        (snapshots && snapshots->numel() > 0 ? static_cast<TY*>(snapshots->data_ptr()) : nullptr),                 \
        static_cast<const int32_t*>(indices.data_ptr()),                                                           \
        static_cast<float*>(correction.data_ptr()),                                                                \
        static_cast<float*>(key.data_ptr()),                                                                       \
        static_cast<float*>(decay.data_ptr()),                                                                     \
        static_cast<bf16_t*>(out.data_ptr()),                                                                      \
        int32_t(H.unwrap()),                                                                                       \
        int32_t(T),                                                                                                \
        int32_t(L.unwrap()),                                                                                       \
        int32_t(S.unwrap()),                                                                                       \
        q1.unwrap(),                                                                                               \
        k1.unwrap(),                                                                                               \
        v1.unwrap(),                                                                                               \
        a1.unwrap(),                                                                                               \
        float(lower)};                                                                                             \
    LaunchKernel(dim3(8, H.unwrap(), B.unwrap()), dim3(32, 4), device.unwrap())(replay_ppu::verify_kernel<TY>, p); \
  } while (0)
  if (is_type<bf16_t>(state.dtype()))
    VERIFY_TYPE(bf16_t);
  else
    VERIFY_TYPE(float);
#undef VERIFY_TYPE
}


/// Restore accepted checkpoints for all KDA layers in one launch.
void replay_commit_impl(
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView correction,
    tvm::ffi::TensorView key,
    tvm::ffi::TensorView decay,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView accepts,
    const tvm::ffi::TensorView* track_indices,
    const tvm::ffi::TensorView* track_steps) {
  using namespace host;
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  auto Layers = SymbolicSize{"layers"}, S = SymbolicSize{"slots"}, H = SymbolicSize{"heads"};
  auto L = SymbolicSize{"ring_len"}, B = SymbolicSize{"batch"};
  TensorMatcher({Layers, S, H, 128, 128}).with_dtype<bf16_t, float>().with_device(device).verify(state);
  TensorMatcher({Layers, S, H, L, 128})
      .with_dtype<float>()
      .with_device(device)
      .verify(correction)
      .verify(key)
      .verify(decay);
  TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(indices).verify(accepts);
  const auto state_alignment = is_type<bf16_t>(state.dtype()) ? 8 : 16;
  CHECK_HOST(reinterpret_cast<uintptr_t>(state.data_ptr()) % state_alignment == 0) << "KDA state must be aligned";
  for (const auto tensor : {correction, key, decay}) {
    CHECK_HOST(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0) << "KDA compact records must be aligned";
  }
  const int32_t* ti = nullptr;
  const int32_t* ts = nullptr;
  CHECK_HOST((track_indices == nullptr) == (track_steps == nullptr)) << "track tensors must be paired";
  if (track_indices) {
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(*track_indices).verify(*track_steps);
    ti = static_cast<const int32_t*>(track_indices->data_ptr());
    ts = static_cast<const int32_t*>(track_steps->data_ptr());
  }
  if (B.unwrap() == 0 || Layers.unwrap() == 0) return;
#define COMMIT_TYPE(TY)                                                                          \
  LaunchKernel(dim3(4, B.unwrap(), Layers.unwrap() * H.unwrap()), dim3(32, 4), device.unwrap())( \
      replay_ppu::commit_kernel<TY>,                                                             \
      static_cast<TY*>(state.data_ptr()),                                                        \
      static_cast<const float*>(correction.data_ptr()),                                          \
      static_cast<const float*>(key.data_ptr()),                                                 \
      static_cast<const float*>(decay.data_ptr()),                                               \
      static_cast<const int32_t*>(indices.data_ptr()),                                           \
      static_cast<const int32_t*>(accepts.data_ptr()),                                           \
      int(H.unwrap()),                                                                           \
      int(L.unwrap()),                                                                           \
      int(S.unwrap()),                                                                           \
      int64_t(state.stride(0)),                                                                  \
      int64_t(correction.stride(0)),                                                             \
      ti,                                                                                        \
      ts)
  if (is_type<bf16_t>(state.dtype()))
    COMMIT_TYPE(bf16_t);
  else
    COMMIT_TYPE(float);
#undef COMMIT_TYPE
}

/// Compact KDA verification: no persistent or intermediate state writes.
void replay_verify(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView dt,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView correction,
    tvm::ffi::TensorView key,
    tvm::ffi::TensorView decay,
    tvm::ffi::TensorView out,
    double lower) {
  replay_verify_impl(q, k, v, a, b, A, dt, state, indices, correction, key, decay, out, lower, nullptr);
}

/// Test oracle: materialize the same native recurrence snapshots.
void replay_verify_snapshot(
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k,
    tvm::ffi::TensorView v,
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView b,
    tvm::ffi::TensorView A,
    tvm::ffi::TensorView dt,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView correction,
    tvm::ffi::TensorView key,
    tvm::ffi::TensorView decay,
    tvm::ffi::TensorView out,
    double lower,
    tvm::ffi::TensorView snapshots) {
  replay_verify_impl(q, k, v, a, b, A, dt, state, indices, correction, key, decay, out, lower, &snapshots);
}

/// Restore accepted state across all layers.
void replay_commit(
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView correction,
    tvm::ffi::TensorView key,
    tvm::ffi::TensorView decay,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView accepts) {
  replay_commit_impl(state, correction, key, decay, indices, accepts, nullptr, nullptr);
}

/// Restore accepted state and an optional prefix-cache crossing checkpoint.
void replay_commit_track(
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView correction,
    tvm::ffi::TensorView key,
    tvm::ffi::TensorView decay,
    tvm::ffi::TensorView indices,
    tvm::ffi::TensorView accepts,
    tvm::ffi::TensorView tracks,
    tvm::ffi::TensorView steps) {
  replay_commit_impl(state, correction, key, decay, indices, accepts, &tracks, &steps);
}
}  // namespace sglang
