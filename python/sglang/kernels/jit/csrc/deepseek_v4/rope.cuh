#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using DType = bf16_t;
constexpr int64_t kRopeDim = 64;
constexpr uint32_t kBlockSize = 128;
constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;

struct FusedQKRopeParams {
  void* __restrict__ q;
  void* __restrict__ k;
  const float* __restrict__ freqs_cis;
  const void* __restrict__ positions;
  int64_t q_stride_batch;
  int64_t k_stride_batch;
  int64_t q_stride_head;
  int64_t k_stride_head;
  uint32_t num_q_heads;
  uint32_t num_k_heads;
  uint32_t batch_size;
};

template <bool kUsePDL, bool kInverse, typename IndexType>
__global__ __launch_bounds__(kBlockSize, 16)  //
    void deepseek_rope_kernel(const __grid_constant__ FusedQKRopeParams param) {
  using namespace device;
  using DType2 = packed_t<DType>;

  const auto warp_id = threadIdx.x / kWarpThreads;
  const auto lane_id = threadIdx.x % kWarpThreads;
  const auto global_warp_id = blockIdx.x * kNumWarps + warp_id;

  const auto& [
    q, k, freqs_cis, positions, //
    q_stride_batch, k_stride_batch, q_stride_head, k_stride_head, //
    num_q_heads, num_k_heads, batch_size
  ] = param;

  const auto num_total_heads = num_q_heads + num_k_heads;
  const auto head_id = global_warp_id % num_total_heads;
  const auto batch_id = global_warp_id / num_total_heads;
  if (batch_id >= batch_size) return;

  const auto position = static_cast<const IndexType*>(positions)[batch_id];
  const auto is_q = head_id < num_q_heads;
  const auto local_head = is_q ? head_id : (head_id - num_q_heads);
  const auto stride_batch = is_q ? q_stride_batch : k_stride_batch;
  const auto stride_head = is_q ? q_stride_head : k_stride_head;
  const auto base_ptr = is_q ? q : k;
  const auto input = static_cast<DType2*>(pointer::offset(base_ptr, batch_id * stride_batch, local_head * stride_head));

  const auto freq_ptr = reinterpret_cast<const fp32x2_t*>(freqs_cis + position * kRopeDim);
  const auto [f_real, f_imag] = freq_ptr[lane_id];
  PDLWaitPrimary<kUsePDL>();

  const auto data = input[lane_id];
  const auto [x_real, x_imag] = cast<fp32x2_t>(data);
  fp32x2_t output;
  if constexpr (kInverse) {
    // (a + bi) * (c - di) = (ac + bd) + (bc - ad)i
    output = {
        x_real * f_real + x_imag * f_imag,
        x_imag * f_real - x_real * f_imag,
    };
  } else {
    // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    output = {
        x_real * f_real - x_imag * f_imag,
        x_real * f_imag + x_imag * f_real,
    };
  }
  input[lane_id] = cast<DType2>(output);

  PDLTriggerSecondary<kUsePDL>();
}

#if defined(SGL_DSV4_Q_ONLY_ROPE_FREQ_REUSE) && \
    (defined(SGL_SAIL) || defined(__HGGCCC__))
template <bool kInverse>
__device__ __forceinline__ packed_t<DType> rotate_rope_pair(
    packed_t<DType> data, float f_real, float f_imag) {
  using namespace device;
  using DType2 = packed_t<DType>;
  const auto [x_real, x_imag] = cast<fp32x2_t>(data);
  fp32x2_t output;
  if constexpr (kInverse) {
    // (a + bi) * (c - di) = (ac + bd) + (bc - ad)i
    output = {
        x_real * f_real + x_imag * f_imag,
        x_imag * f_real - x_real * f_imag,
    };
  } else {
    // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    output = {
        x_real * f_real - x_imag * f_imag,
        x_real * f_imag + x_imag * f_real,
    };
  }
  return cast<DType2>(output);
}

template <bool kUsePDL, bool kInverse, typename IndexType, unsigned Heads, unsigned Group>
__global__ __launch_bounds__(kBlockSize, 16)
void deepseek_rope_q_only_kernel(const __grid_constant__ FusedQKRopeParams param) {
  using namespace device;
  using DType2 = packed_t<DType>;
  constexpr unsigned heads_per_block = kNumWarps * Group;
  static_assert(Heads % heads_per_block == 0);
  constexpr unsigned blocks_per_token = Heads / heads_per_block;

  const auto warp = threadIdx.x / kWarpThreads;
  const auto lane = threadIdx.x % kWarpThreads;
  const auto token = blockIdx.x / blocks_per_token;
  const auto head = (blockIdx.x % blocks_per_token) * heads_per_block + warp;
  const auto position = static_cast<const IndexType*>(param.positions)[token];
  const auto [f_real, f_imag] = reinterpret_cast<const fp32x2_t*>(param.freqs_cis + position * kRopeDim)[lane];
  auto* row = pointer::offset(param.q, token * param.q_stride_batch, head * param.q_stride_head);
  PDLWaitPrimary<kUsePDL>();
  #pragma unroll
  for (unsigned i = 0; i < Group; ++i) {
    auto* head_pair = static_cast<DType2*>(pointer::offset(row, i * kNumWarps * param.q_stride_head));
    head_pair[lane] = rotate_rope_pair<kInverse>(head_pair[lane], f_real, f_imag);
  }
  PDLTriggerSecondary<kUsePDL>();
}

#endif

template <bool kUsePDL>
struct FusedQKRopeKernel {
  // 4 kernel variants: {forward, inverse} x {int32, int64}
  static constexpr auto kernel_fwd_i32 = deepseek_rope_kernel<kUsePDL, false, int32_t>;
  static constexpr auto kernel_fwd_i64 = deepseek_rope_kernel<kUsePDL, false, int64_t>;
  static constexpr auto kernel_inv_i32 = deepseek_rope_kernel<kUsePDL, true, int32_t>;
  static constexpr auto kernel_inv_i64 = deepseek_rope_kernel<kUsePDL, true, int64_t>;


#if defined(SGL_DSV4_Q_ONLY_ROPE_FREQ_REUSE) && \
    (defined(SGL_SAIL) || defined(__HGGCCC__))
  template <unsigned Heads, unsigned Group>
  static void launch_q_only(
      FusedQKRopeParams params, bool pos_i32, bool inverse, DLDevice device) {
    constexpr auto fwd32 = deepseek_rope_q_only_kernel<kUsePDL, false, int32_t, Heads, Group>;
    constexpr auto fwd64 = deepseek_rope_q_only_kernel<kUsePDL, false, int64_t, Heads, Group>;
    constexpr auto inv32 = deepseek_rope_q_only_kernel<kUsePDL, true, int32_t, Heads, Group>;
    constexpr auto inv64 = deepseek_rope_q_only_kernel<kUsePDL, true, int64_t, Heads, Group>;
    const auto kernel = inverse ? (pos_i32 ? inv32 : inv64) : (pos_i32 ? fwd32 : fwd64);
    const auto blocks = params.batch_size * (Heads / (kNumWarps * Group));
    host::LaunchKernel(blocks, kBlockSize, device).enable_pdl(kUsePDL)(kernel, params);
  }
#endif

  static void forward(
      const tvm::ffi::TensorView q,
      const tvm::ffi::Optional<tvm::ffi::TensorView> k,
      const tvm::ffi::TensorView freqs_cis,
      const tvm::ffi::TensorView positions,
      bool inverse) {
    using namespace host;

    auto B = SymbolicSize{"batch_size"};
    auto Q = SymbolicSize{"num_q_heads"};
    auto K = SymbolicSize{"num_k_heads"};
    constexpr auto D = kRopeDim;
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({B, Q, D})  //
        .with_strides({-1, -1, 1})
        .with_dtype<DType>()
        .with_device(device_)
        .verify(q);
    if (k.has_value()) {
      TensorMatcher({B, K, D})  //
          .with_strides({-1, -1, 1})
          .with_dtype<DType>()
          .with_device(device_)
          .verify(k.value());
    } else {
      K.set_value(0);
    }
    TensorMatcher({-1, D})  //
        .with_dtype<float>()
        .with_device(device_)
        .verify(freqs_cis);

    auto pos_dtype = SymbolicDType{};
    TensorMatcher({B})  //
        .with_dtype<int32_t, int64_t>(pos_dtype)
        .with_device(device_)
        .verify(positions);
    const bool pos_i32 = pos_dtype.is_type<int32_t>();

    const auto batch_size = static_cast<uint32_t>(B.unwrap());
    if (batch_size == 0) return;

    const auto num_q_heads = static_cast<uint32_t>(Q.unwrap());
    const auto num_k_heads = static_cast<uint32_t>(K.unwrap());
    const auto num_total_heads = num_q_heads + num_k_heads;
    const auto total_warps = batch_size * num_total_heads;
    const auto num_blocks = div_ceil(total_warps, kNumWarps);

    const auto elem_size = static_cast<int64_t>(sizeof(DType));
    const auto params = FusedQKRopeParams{
        .q = q.data_ptr(),
        .k = k ? k.value().data_ptr() : nullptr,
        .freqs_cis = static_cast<const float*>(freqs_cis.data_ptr()),
        .positions = positions.data_ptr(),
        .q_stride_batch = q.stride(0) * elem_size,
        .k_stride_batch = k ? k.value().stride(0) * elem_size : 0,
        .q_stride_head = q.stride(1) * elem_size,
        .k_stride_head = k ? k.value().stride(1) * elem_size : 0,
        .num_q_heads = num_q_heads,
        .num_k_heads = num_k_heads,
        .batch_size = batch_size,
    };

#if defined(SGL_DSV4_Q_ONLY_ROPE_FREQ_REUSE) && \
    (defined(SGL_SAIL) || defined(__HGGCCC__))
    if (num_k_heads == 0 && batch_size >= 256) {
      switch (num_q_heads) {
        case 8: return launch_q_only<8, 1>(params, pos_i32, inverse, device_.unwrap());
        case 16: return launch_q_only<16, 4>(params, pos_i32, inverse, device_.unwrap());
        case 32: return launch_q_only<32, 8>(params, pos_i32, inverse, device_.unwrap());
        case 64: return launch_q_only<64, 16>(params, pos_i32, inverse, device_.unwrap());
        case 128: return launch_q_only<128, 16>(params, pos_i32, inverse, device_.unwrap());
      }
    }
#endif

    // dispatch: {inverse} x {pos_i32}
    using KernelType = decltype(kernel_fwd_i32);
    const KernelType kernel =
        inverse ? (pos_i32 ? kernel_inv_i32 : kernel_inv_i64) : (pos_i32 ? kernel_fwd_i32 : kernel_fwd_i64);
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap())  //
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
