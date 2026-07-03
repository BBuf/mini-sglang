#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace {

struct RouterGemvParams {
  const void *__restrict__ x;  // [M, K] bf16
  const void *__restrict__ w;  // [E, K] bf16 (row-major expert weights)
  void *__restrict__ out;      // [M, E] fp32 logits
  std::size_t x_stride;        // elements between x rows
  std::size_t num_tokens;      // M
  std::size_t num_experts;     // E
  std::size_t hidden;          // K
};

// One block per expert; every thread strides over K with 8-wide bf16 vectors
// and keeps one fp32 accumulator per token row (decode M is tiny). Products
// are computed in fp32 from exactly-representable bf16 inputs, so the result
// matches an fp32 GEMM of pre-cast operands up to accumulation order.
template <std::size_t kNumThreads, std::size_t kMaxM, bool kUsePDL>
__global__ __launch_bounds__(kNumThreads) void //
    router_gemv_kernel(const __grid_constant__ RouterGemvParams params) {
  const auto &[xp, wp, outp, x_stride, M, E, K] = params;
  const auto e = blockIdx.x;
  if (e >= E) return;

  const auto *x = static_cast<const __nv_bfloat16 *>(xp);
  const auto *w = static_cast<const __nv_bfloat16 *>(wp) + e * K;
  auto *out = static_cast<float *>(outp);

  device::PDL::wait<kUsePDL>();

  float acc[kMaxM];
#pragma unroll
  for (std::size_t m = 0; m < kMaxM; ++m) acc[m] = 0.0f;

  constexpr std::size_t kVec = 8;  // 8 x bf16 = 16 bytes
  const auto tid = threadIdx.x;
  for (std::size_t k = tid * kVec; k < K; k += kNumThreads * kVec) {
    const auto wv = *reinterpret_cast<const uint4 *>(w + k);
    const auto *wh = reinterpret_cast<const __nv_bfloat16 *>(&wv);
    for (std::size_t m = 0; m < M; ++m) {
      const auto xv = *reinterpret_cast<const uint4 *>(x + m * x_stride + k);
      const auto *xh = reinterpret_cast<const __nv_bfloat16 *>(&xv);
      float s = 0.0f;
#pragma unroll
      for (std::size_t i = 0; i < kVec; ++i) {
        s += __bfloat162float(xh[i]) * __bfloat162float(wh[i]);
      }
      acc[m] += s;
    }
  }

  // intra-warp reduce, then cross-warp reduce through shared memory
  constexpr std::size_t kWarps = kNumThreads / 32;
  __shared__ float smem[kWarps][kMaxM];
  const auto lane = tid % 32;
  const auto warp = tid / 32;
#pragma unroll
  for (std::size_t m = 0; m < kMaxM; ++m) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      acc[m] += __shfl_down_sync(0xffffffff, acc[m], off);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (std::size_t m = 0; m < kMaxM; ++m) smem[warp][m] = acc[m];
  }
  __syncthreads();
  if (warp == 0 && lane < kWarps) {
    constexpr unsigned kMask = (kWarps >= 32) ? 0xffffffffu : ((1u << kWarps) - 1u);
    for (std::size_t m = 0; m < M; ++m) {
      float v = smem[lane][m];
#pragma unroll
      for (unsigned off = kWarps / 2; off > 0; off >>= 1) {
        v += __shfl_down_sync(kMask, v, off);
      }
      if (lane == 0) out[m * E + e] = v;
    }
  }

  device::PDL::launch<kUsePDL>();
}

template <std::size_t max_m = 8, std::size_t num_threads = 128, bool use_pdl = false>
struct RouterGemvKernel {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView w,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto M = SymbolicSize{"M"};
    auto K = SymbolicSize{"K"};
    auto E = SymbolicSize{"E"};
    auto Xs = SymbolicSize{"Xs"};
    auto device_ = SymbolicDevice{};
    auto in_dtype = SymbolicDType{};

    TensorMatcher({M, K})
        .with_strides({Xs, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(in_dtype)
        .verify(x);
    TensorMatcher({E, K})
        .with_device<kDLCUDA>(device_)
        .with_dtype(in_dtype)
        .verify(w);
    TensorMatcher({M, E})
        .with_device<kDLCUDA>(device_)
        .with_dtype<float>()
        .verify(out);

    // dtype_trait has no bf16 mapping; check the DLPack code by hand
    const auto in_dt = in_dtype.unwrap();
    RuntimeCheck(in_dt.code == DLDataTypeCode::kDLBfloat && in_dt.bits == 16);
    RuntimeCheck(static_cast<std::size_t>(M.unwrap()) <= max_m);
    RuntimeCheck(K.unwrap() % (8 * num_threads) == 0);

    const auto params = RouterGemvParams{
        .x = x.data_ptr(),
        .w = w.data_ptr(),
        .out = out.data_ptr(),
        .x_stride = static_cast<std::size_t>(Xs.unwrap()),
        .num_tokens = static_cast<std::size_t>(M.unwrap()),
        .num_experts = static_cast<std::size_t>(E.unwrap()),
        .hidden = static_cast<std::size_t>(K.unwrap()),
    };
    LaunchKernel(E.unwrap(), num_threads, device_.unwrap())
        .with_attr(use_pdl)(router_gemv_kernel<num_threads, max_m, use_pdl>, params);
  }
};

}  // namespace
