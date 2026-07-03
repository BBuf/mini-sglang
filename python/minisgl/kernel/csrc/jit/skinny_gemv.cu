#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace {

struct SkinnyGemvParams {
  const void *__restrict__ x;  // [M, K] bf16
  const void *__restrict__ w;  // [N, K] bf16 (row-major; out = x @ w^T)
  void *__restrict__ out;      // [M, N] bf16
  std::size_t x_stride;        // elements between x rows
  std::size_t out_stride;      // elements between out rows
  std::size_t num_tokens;      // M
  std::size_t num_rows;        // N
  std::size_t hidden;          // K
};

// Decode-shape (M <= kMaxM) GEMV: each block owns kRows output columns (rows of
// W); threads stride over K with 8-wide bf16 vectors keeping one fp32
// accumulator per (row, token). Replaces cuBLAS splitK GEMM + splitKreduce
// pairs that dominate skinny bf16 GEMMs at M ~ 4.
template <std::size_t kNumThreads, std::size_t kMaxM, std::size_t kRows, bool kUsePDL>
__global__ __launch_bounds__(kNumThreads) void //
    skinny_gemv_kernel(const __grid_constant__ SkinnyGemvParams params) {
  const auto &[xp, wp, outp, x_stride, out_stride, M, N, K] = params;
  const auto row0 = blockIdx.x * kRows;
  if (row0 >= N) return;

  const auto *x = static_cast<const __nv_bfloat16 *>(xp);
  const auto *w = static_cast<const __nv_bfloat16 *>(wp);
  auto *out = static_cast<__nv_bfloat16 *>(outp);

  device::PDL::wait<kUsePDL>();

  constexpr std::size_t kVec = 8;  // 8 x bf16 = 16 bytes
  const auto tid = threadIdx.x;

  float acc[kRows][kMaxM];
#pragma unroll
  for (std::size_t r = 0; r < kRows; ++r)
#pragma unroll
    for (std::size_t m = 0; m < kMaxM; ++m) acc[r][m] = 0.0f;

  for (std::size_t k = tid * kVec; k < K; k += kNumThreads * kVec) {
    __nv_bfloat16 xv[kMaxM][kVec];
    for (std::size_t m = 0; m < M; ++m) {
      *reinterpret_cast<uint4 *>(xv[m]) =
          *reinterpret_cast<const uint4 *>(x + m * x_stride + k);
    }
#pragma unroll
    for (std::size_t r = 0; r < kRows; ++r) {
      const auto row = row0 + r;
      if (row >= N) break;
      const auto wv = *reinterpret_cast<const uint4 *>(w + row * K + k);
      const auto *wh = reinterpret_cast<const __nv_bfloat16 *>(&wv);
      for (std::size_t m = 0; m < M; ++m) {
        float s = 0.0f;
#pragma unroll
        for (std::size_t i = 0; i < kVec; ++i) {
          s += __bfloat162float(xv[m][i]) * __bfloat162float(wh[i]);
        }
        acc[r][m] += s;
      }
    }
  }

  // intra-warp then cross-warp reduction per (row, token)
  constexpr std::size_t kWarps = kNumThreads / 32;
  __shared__ float smem[kWarps][kRows][kMaxM];
  const auto lane = tid % 32;
  const auto warp = tid / 32;
#pragma unroll
  for (std::size_t r = 0; r < kRows; ++r)
#pragma unroll
    for (std::size_t m = 0; m < kMaxM; ++m)
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        acc[r][m] += __shfl_down_sync(0xffffffffu, acc[r][m], off);
  if (lane == 0) {
#pragma unroll
    for (std::size_t r = 0; r < kRows; ++r)
#pragma unroll
      for (std::size_t m = 0; m < kMaxM; ++m) smem[warp][r][m] = acc[r][m];
  }
  __syncthreads();
  if (warp == 0 && lane < kWarps) {
    constexpr unsigned kMask = (kWarps >= 32) ? 0xffffffffu : ((1u << kWarps) - 1u);
#pragma unroll
    for (std::size_t r = 0; r < kRows; ++r) {
      const auto row = row0 + r;
      if (row >= N) break;
      for (std::size_t m = 0; m < M; ++m) {
        float v = smem[lane][r][m];
#pragma unroll
        for (unsigned off = kWarps / 2; off > 0; off >>= 1)
          v += __shfl_down_sync(kMask, v, off);
        if (lane == 0) out[m * out_stride + row] = __float2bfloat16(v);
      }
    }
  }

  device::PDL::launch<kUsePDL>();
}

template <std::size_t rows_per_block, std::size_t num_threads = 128,
          std::size_t max_m = 8, bool use_pdl = false>
struct SkinnyGemvKernel {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView w,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto M = SymbolicSize{"M"};
    auto K = SymbolicSize{"K"};
    auto N = SymbolicSize{"N"};
    auto Xs = SymbolicSize{"Xs"};
    auto Os = SymbolicSize{"Os"};
    auto device_ = SymbolicDevice{};
    auto in_dtype = SymbolicDType{};

    TensorMatcher({M, K})
        .with_strides({Xs, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(in_dtype)
        .verify(x);
    TensorMatcher({N, K})
        .with_device<kDLCUDA>(device_)
        .with_dtype(in_dtype)
        .verify(w);
    TensorMatcher({M, N})
        .with_strides({Os, 1})
        .with_device<kDLCUDA>(device_)
        .with_dtype(in_dtype)
        .verify(out);

    const auto in_dt = in_dtype.unwrap();
    RuntimeCheck(in_dt.code == DLDataTypeCode::kDLBfloat && in_dt.bits == 16);
    RuntimeCheck(static_cast<std::size_t>(M.unwrap()) <= max_m);
    RuntimeCheck(K.unwrap() % 8 == 0);

    const auto params = SkinnyGemvParams{
        .x = x.data_ptr(),
        .w = w.data_ptr(),
        .out = out.data_ptr(),
        .x_stride = static_cast<std::size_t>(Xs.unwrap()),
        .out_stride = static_cast<std::size_t>(Os.unwrap()),
        .num_tokens = static_cast<std::size_t>(M.unwrap()),
        .num_rows = static_cast<std::size_t>(N.unwrap()),
        .hidden = static_cast<std::size_t>(K.unwrap()),
    };
    const auto blocks = (N.unwrap() + rows_per_block - 1) / rows_per_block;
    LaunchKernel(blocks, num_threads, device_.unwrap()).with_attr(use_pdl)(
        skinny_gemv_kernel<num_threads, max_m, rows_per_block, use_pdl>, params);
  }
};

}  // namespace
