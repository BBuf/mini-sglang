#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>
#include <minisgl/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <concepts>
#include <cstddef>
#include <cstdint>

namespace {

struct StoreMLAKernelParams {
  void *__restrict__ ckv_cache;
  void *__restrict__ kpe_cache;
  const void *__restrict__ indices;
  const void *__restrict__ ckv;
  const void *__restrict__ kpe;
  std::size_t ckv_cache_stride;
  std::size_t kpe_cache_stride;
  std::size_t ckv_input_stride;
  std::size_t kpe_input_stride;
  std::size_t length;
};

// Latent-MLA variant of store.cu: the two rows written per token have different
// widths (ckv = kv_lora_rank, k_pe = qk_rope_head_dim), which StoreKernel's
// single element-size template cannot express.
template <std::size_t kNumThreads, std::size_t kMaxOccupancy, bool kUsePDL,
          std::size_t kCkvSize, std::size_t kKpeSize, std::integral T>
__global__ __launch_bounds__(kNumThreads, kMaxOccupancy) void //
    store_mla_kv_cache(const __grid_constant__ StoreMLAKernelParams params) {
  using namespace device;

  constexpr auto kWarpPerBlock =
      static_cast<unsigned>(kNumThreads / kWarpThreads);
  static_assert(kNumThreads % kWarpThreads == 0);

  const auto &[ckv_cache, kpe_cache, indices, ckv, kpe, ckv_cache_stride,
               kpe_cache_stride, ckv_input_stride, kpe_input_stride, length] =
      params;
  const auto warp_id =
      (threadIdx.x / kWarpThreads) + blockIdx.x * kWarpPerBlock;
  PDL::wait<kUsePDL>();

  // each warp stores one token's compressed latent + rope key
  if (warp_id < length) {
    const auto pos = static_cast<const T *>(indices)[warp_id];
    warp::copy<kCkvSize>(pointer::offset(ckv_cache, pos * ckv_cache_stride),
                         pointer::offset(ckv, warp_id * ckv_input_stride));
    warp::copy<kKpeSize>(pointer::offset(kpe_cache, pos * kpe_cache_stride),
                         pointer::offset(kpe, warp_id * kpe_input_stride));
  }

  PDL::launch<kUsePDL>();
}

template <std::size_t ckv_size,       // bytes per ckv row (dtype * kv_lora_rank)
          std::size_t kpe_size,       // bytes per k_pe row (dtype * rope dim)
          std::size_t num_threads = 128,   // number of threads per block
          std::size_t max_concurrency = 1, // max blocks per SM
          bool use_pdl = false>
struct StoreMLAKernel {
  static void run(const tvm::ffi::TensorView ckv_cache,
                  const tvm::ffi::TensorView kpe_cache,
                  const tvm::ffi::TensorView indices,
                  const tvm::ffi::TensorView ckv,
                  const tvm::ffi::TensorView kpe) {
    using namespace host;
    auto Dk = SymbolicSize{"Dk"}; // ckv row elements
    auto Dv = SymbolicSize{"Dv"}; // k_pe row elements
    auto L = SymbolicSize{"L"};   // length
    auto Xk = SymbolicSize{"Xk"}; // stride ckv cache
    auto Xv = SymbolicSize{"Xv"}; // stride kpe cache
    auto Yk = SymbolicSize{"Yk"}; // stride ckv input
    auto Yv = SymbolicSize{"Yv"}; // stride kpe input
    auto indices_dtype_ = SymbolicDType{};
    auto dtype_ = SymbolicDType{};
    auto device_ = SymbolicDevice{};

    TensorMatcher({-1, Dk}) //
        .with_strides({Xk, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(dtype_)
        .verify(ckv_cache);
    TensorMatcher({-1, Dv}) //
        .with_strides({Xv, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(dtype_)
        .verify(kpe_cache);
    TensorMatcher({L, Dk}) //
        .with_strides({Yk, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(dtype_)
        .verify(ckv);
    TensorMatcher({L, Dv}) //
        .with_strides({Yv, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(dtype_)
        .verify(kpe);
    TensorMatcher({L}) //
        .with_device<kDLCUDA>(device_)
        .with_dtype<int32_t, int64_t>(indices_dtype_)
        .verify(indices);

    const auto dtype_size = dtype_bytes(dtype_.unwrap());
    RuntimeCheck(ckv_size == dtype_size * Dk.unwrap());
    RuntimeCheck(kpe_size == dtype_size * Dv.unwrap());

    const auto device = device_.unwrap();
    const auto use_int32 = indices_dtype_.unwrap().bits == 32;
    const auto length = static_cast<std::size_t>(L.unwrap());

    const auto params = StoreMLAKernelParams{
        .ckv_cache = ckv_cache.data_ptr(),
        .kpe_cache = kpe_cache.data_ptr(),
        .indices = indices.data_ptr(),
        .ckv = ckv.data_ptr(),
        .kpe = kpe.data_ptr(),
        .ckv_cache_stride = Xk.unwrap() * dtype_size,
        .kpe_cache_stride = Xv.unwrap() * dtype_size,
        .ckv_input_stride = Yk.unwrap() * dtype_size,
        .kpe_input_stride = Yv.unwrap() * dtype_size,
        .length = length,
    };

    constexpr auto kWarpPerBlock = num_threads / 32;
    static_assert(num_threads % 32 == 0);
    const auto num_blocks = div_ceil(length, kWarpPerBlock);
    const auto kernel =
        use_int32 ? store_mla_kv_cache<num_threads, max_concurrency, use_pdl,
                                       ckv_size, kpe_size, int32_t>
                  : store_mla_kv_cache<num_threads, max_concurrency, use_pdl,
                                       ckv_size, kpe_size, int64_t>;
    LaunchKernel(num_blocks, num_threads, device)
        .with_attr(use_pdl)(kernel, params);
  }
};

} // namespace
