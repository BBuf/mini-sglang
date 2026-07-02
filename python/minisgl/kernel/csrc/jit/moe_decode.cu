#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>

#include <cstddef>
#include <cstdint>

namespace {

// Decode-shape fused MoE for block-fp8 experts (weights fp8 e4m3 with
// [128,128] block scales, passed as uint8 views), bf16 activations kept
// UNQUANTIZED: better numerics than the w8a8 path and none of the
// act-quant / align / sort / moe-sum machinery.
//
// Kernel 1: inter[t,s,j] = silu(x[t].gate[e_s][j]) * (x[t].up[e_s][j])
// Kernel 2: out[t,h]     = sum_s w[t,s] * (inter[t,s,:].down[e_s][h,:])

__device__ __forceinline__ float2 fp8x2_to_float2(std::uint16_t b2) {
  const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(b2, __NV_E4M3);
  return __half22float2(*reinterpret_cast<const __half2 *>(&hr));
}

struct MoeG1Params {
  const void *__restrict__ x;        // [M, H] bf16
  const void *__restrict__ gate_up;  // [E, 2I, H] fp8 (uint8 view)
  const void *__restrict__ gu_scale; // [E, 2I/128, H/128] fp32
  const void *__restrict__ topk_ids; // [M, K] int32
  void *__restrict__ inter;          // [M, K, I] bf16
  std::size_t M, K, H, I;
};

template <std::size_t kThreads, std::size_t kSliceI>
__global__ __launch_bounds__(kThreads) void moe_g1_kernel(
    const __grid_constant__ MoeG1Params p) {
  const auto pair = blockIdx.x;  // t * K + slot
  const auto t = pair / p.K;
  const auto e = static_cast<std::size_t>(
      static_cast<const std::int32_t *>(p.topk_ids)[pair]);

  const auto *w = static_cast<const std::uint8_t *>(p.gate_up) + e * 2 * p.I * p.H;
  const auto *sc =
      static_cast<const float *>(p.gu_scale) + e * (2 * p.I / 128) * (p.H / 128);
  auto *inter = static_cast<__nv_bfloat16 *>(p.inter) + pair * p.I;

  constexpr std::size_t kWarps = kThreads / 32;
  constexpr std::size_t kChunk = 256;  // bytes of each row staged per step
  static_assert(kSliceI == 2 * kWarps, "each warp owns exactly two j");

  // smem: fp32 activation row (24KB) + per-warp cp.async double buffer of the
  // four weight rows' current/next 256B chunks (plain LDG left the DRAM pipe
  // underfilled; async staging runs copy and math concurrently)
  extern __shared__ float xs[];
  auto *stage = reinterpret_cast<std::uint8_t *>(xs + p.H) +
                (threadIdx.x / 32) * 2 * 4 * kChunk;

  for (std::size_t i = threadIdx.x; i < p.H; i += kThreads)
    xs[i] = __bfloat162float(static_cast<const __nv_bfloat16 *>(p.x)[t * p.H + i]);

  const auto warp = threadIdx.x / 32;
  const auto lane = threadIdx.x % 32;
  const auto j0 = blockIdx.y * kSliceI;
  const auto sc_cols = p.H / 128;

  const auto ja = j0 + warp;
  const auto jb = ja + kWarps;
  const std::uint8_t *rows[4] = {w + ja * p.H, w + (ja + p.I) * p.H,
                                 w + jb * p.H, w + (jb + p.I) * p.H};
  const float *scs[4] = {sc + (ja >> 7) * sc_cols, sc + ((ja + p.I) >> 7) * sc_cols,
                         sc + (jb >> 7) * sc_cols, sc + ((jb + p.I) >> 7) * sc_cols};

  const auto issue = [&](std::size_t cb) {
    auto *dst = stage + ((cb / kChunk) & 1) * 4 * kChunk;
    const auto half = lane / 16;         // lanes 0-15 rows 0/1, 16-31 rows 2/3
    const auto sub = (lane % 16) * 16;   // 16 lanes x 16B == 256B chunk
#pragma unroll
    for (int r = 0; r < 2; ++r)
      __pipeline_memcpy_async(dst + (2 * half + r) * kChunk + sub,
                              rows[2 * half + r] + cb + sub, 16);
    __pipeline_commit();
  };
  issue(0);

  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  __syncthreads();  // xs ready
  for (std::size_t cb = 0; cb < p.H; cb += kChunk) {
    if (cb + kChunk < p.H) {
      issue(cb + kChunk);
      __pipeline_wait_prior(1);
    } else {
      __pipeline_wait_prior(0);
    }
    __syncwarp();
    const auto *sb = stage + ((cb / kChunk) & 1) * 4 * kChunk;
    const auto kk = lane * 8;  // 32 lanes x 8B == 256B chunk
    const float4 xv0 = *reinterpret_cast<const float4 *>(xs + cb + kk);
    const float4 xv1 = *reinterpret_cast<const float4 *>(xs + cb + kk + 4);
    const float sblk = static_cast<float>((cb + kk) >> 7);
    (void)sblk;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const uint2 w8 = *reinterpret_cast<const uint2 *>(sb + r * kChunk + kk);
      const float2 f0 = fp8x2_to_float2(w8.x & 0xffffu);
      const float2 f1 = fp8x2_to_float2(w8.x >> 16);
      const float2 f2 = fp8x2_to_float2(w8.y & 0xffffu);
      const float2 f3 = fp8x2_to_float2(w8.y >> 16);
      float part = xv0.x * f0.x + xv0.y * f0.y + xv0.z * f1.x + xv0.w * f1.y;
      part += xv1.x * f2.x + xv1.y * f2.y + xv1.z * f3.x + xv1.w * f3.y;
      acc[r] += part * scs[r][(cb + kk) >> 7];
    }
    __syncwarp();  // all lanes done with this buffer before refill at cb + 2
  }
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
  if (lane == 0) {
    const float ga = acc[0] / (1.0f + expf(-acc[0]));
    const float gb = acc[2] / (1.0f + expf(-acc[2]));
    inter[ja] = __float2bfloat16(ga * acc[1]);
    inter[jb] = __float2bfloat16(gb * acc[3]);
  }
}

struct MoeG2Params {
  const void *__restrict__ inter;    // [M, K, I] bf16
  const void *__restrict__ down;     // [E, H, I] fp8 (uint8 view)
  const void *__restrict__ dn_scale; // [E, H/128, I/128] fp32
  const void *__restrict__ topk_ids; // [M, K] int32
  const void *__restrict__ topk_w;   // [M, K] float32
  void *__restrict__ out;            // [M, H] bf16
  std::size_t M, K, H, I;
};

template <std::size_t kThreads, std::size_t kSliceH>
__global__ __launch_bounds__(kThreads) void moe_g2_kernel(
    const __grid_constant__ MoeG2Params p) {
  const auto t = blockIdx.x;
  const auto h0 = blockIdx.y * kSliceH;

  // smem: this token's K inter vectors, then a double buffer for the down-row
  // tile of the current/next expert (cp.async keeps the DRAM pipe full: the
  // plain-LDG version stalled at ~1.6TB/s from too little memory-level
  // parallelism)
  extern __shared__ std::uint8_t smem_raw[];
  auto *is = reinterpret_cast<__nv_bfloat16 *>(smem_raw);
  auto *stage = smem_raw + p.K * p.I * sizeof(__nv_bfloat16);
  const auto tile = kSliceH * p.I;

  for (std::size_t i = threadIdx.x; i < p.K * p.I; i += kThreads)
    is[i] = static_cast<const __nv_bfloat16 *>(p.inter)[t * p.K * p.I + i];

  const auto *ids = static_cast<const std::int32_t *>(p.topk_ids) + t * p.K;
  const auto *ws = static_cast<const float *>(p.topk_w) + t * p.K;
  const auto sc_cols = p.I / 128;

  const auto issue = [&](std::size_t s) {
    const auto e = static_cast<std::size_t>(ids[s]);
    const auto *src =
        static_cast<const std::uint8_t *>(p.down) + (e * p.H + h0) * p.I;
    auto *dst = stage + (s & 1) * tile;
    for (std::size_t off = threadIdx.x * 16; off < tile; off += kThreads * 16)
      __pipeline_memcpy_async(dst + off, src + off, 16);
    __pipeline_commit();
  };
  issue(0);

  constexpr std::size_t kWarps = kThreads / 32;
  constexpr std::size_t kHPerWarp = kSliceH / kWarps;
  const auto warp = threadIdx.x / 32;
  const auto lane = threadIdx.x % 32;

  float acc[kHPerWarp];
#pragma unroll
  for (std::size_t r = 0; r < kHPerWarp; ++r) acc[r] = 0.0f;
  const auto kk = lane * 8;  // I == 256: each lane owns one 8-byte chunk

  for (std::size_t s = 0; s < p.K; ++s) {
    if (s + 1 < p.K) issue(s + 1);
    if (s + 1 < p.K) {
      __pipeline_wait_prior(1);
    } else {
      __pipeline_wait_prior(0);
    }
    __syncthreads();  // staged tile (and, on s==0, `is`) visible to all
    const auto e = static_cast<std::size_t>(ids[s]);
    const float sd = static_cast<const float *>(p.dn_scale)[
        (e * (p.H / 128) + (h0 >> 7)) * sc_cols + (kk >> 7)] * ws[s];
    const auto *sb = stage + (s & 1) * tile + warp * kHPerWarp * p.I + kk;
    const auto *iv = is + s * p.I + kk;
    float xr[8];
#pragma unroll
    for (int v = 0; v < 8; ++v) xr[v] = __bfloat162float(iv[v]);
#pragma unroll
    for (std::size_t r = 0; r < kHPerWarp; ++r) {
      const uint2 w8 = *reinterpret_cast<const uint2 *>(sb + r * p.I);
      const float2 f01 = fp8x2_to_float2(w8.x & 0xffffu);
      const float2 f23 = fp8x2_to_float2(w8.x >> 16);
      const float2 f45 = fp8x2_to_float2(w8.y & 0xffffu);
      const float2 f67 = fp8x2_to_float2(w8.y >> 16);
      float part = xr[0] * f01.x + xr[1] * f01.y + xr[2] * f23.x + xr[3] * f23.y +
                   xr[4] * f45.x + xr[5] * f45.y + xr[6] * f67.x + xr[7] * f67.y;
      acc[r] += part * sd;
    }
    __syncthreads();  // compute done before this buffer is refilled at s + 2
  }
#pragma unroll
  for (std::size_t r = 0; r < kHPerWarp; ++r) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      acc[r] += __shfl_down_sync(0xffffffffu, acc[r], off);
    if (lane == 0)
      static_cast<__nv_bfloat16 *>(p.out)[t * p.H + h0 + warp * kHPerWarp + r] =
          __float2bfloat16(acc[r]);
  }
}

template <std::size_t threads1 = 256, std::size_t slice_i = 16,
          std::size_t threads2 = 128, std::size_t slice_h = 64>
struct MoeDecodeKernel {
  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView gate_up,
                  const tvm::ffi::TensorView gu_scale, const tvm::ffi::TensorView down,
                  const tvm::ffi::TensorView dn_scale, const tvm::ffi::TensorView topk_ids,
                  const tvm::ffi::TensorView topk_w, const tvm::ffi::TensorView inter,
                  const tvm::ffi::TensorView out) {
    using namespace host;
    auto M = SymbolicSize{"M"};
    auto H = SymbolicSize{"H"};
    auto I = SymbolicSize{"I"};
    auto E = SymbolicSize{"E"};
    auto Ktop = SymbolicSize{"Ktop"};
    auto I2 = SymbolicSize{"I2"};
    auto device_ = SymbolicDevice{};
    auto bf = SymbolicDType{};
    auto u8 = SymbolicDType{};
    auto f32 = SymbolicDType{};
    auto i32 = SymbolicDType{};

    TensorMatcher({M, H}).with_device<kDLCUDA>(device_).with_dtype(bf).verify(x);
    TensorMatcher({E, I2, H}).with_device<kDLCUDA>(device_).with_dtype(u8).verify(gate_up);
    TensorMatcher({E, H, I}).with_device<kDLCUDA>(device_).with_dtype(u8).verify(down);
    TensorMatcher({M, Ktop}).with_device<kDLCUDA>(device_).with_dtype(i32).verify(topk_ids);
    TensorMatcher({M, Ktop}).with_device<kDLCUDA>(device_).with_dtype(f32).verify(topk_w);
    TensorMatcher({M, Ktop, I}).with_device<kDLCUDA>(device_).with_dtype(bf).verify(inter);
    TensorMatcher({M, H}).with_device<kDLCUDA>(device_).with_dtype(bf).verify(out);

    const auto bf_dt = bf.unwrap();
    RuntimeCheck(bf_dt.code == DLDataTypeCode::kDLBfloat && bf_dt.bits == 16);
    const auto u8_dt = u8.unwrap();
    RuntimeCheck(u8_dt.code == DLDataTypeCode::kDLUInt && u8_dt.bits == 8);

    const auto m = static_cast<std::size_t>(M.unwrap());
    const auto h = static_cast<std::size_t>(H.unwrap());
    const auto i = static_cast<std::size_t>(I.unwrap());
    const auto k = static_cast<std::size_t>(Ktop.unwrap());
    RuntimeCheck(static_cast<std::size_t>(I2.unwrap()) == 2 * i);
    RuntimeCheck(h % 256 == 0 && i % 256 == 0);
    RuntimeCheck(i % slice_i == 0 && h % slice_h == 0);

    const auto dev = device_.unwrap();
    const auto g1 = MoeG1Params{
        .x = x.data_ptr(),
        .gate_up = gate_up.data_ptr(),
        .gu_scale = gu_scale.data_ptr(),
        .topk_ids = topk_ids.data_ptr(),
        .inter = inter.data_ptr(),
        .M = m, .K = k, .H = h, .I = i,
    };
    LaunchKernel(dim3(m * k, i / slice_i), threads1, dev,
                 h * sizeof(float) + (threads1 / 32) * 2 * 4 * 256)(moe_g1_kernel<threads1, slice_i>, g1);

    const auto g2 = MoeG2Params{
        .inter = inter.data_ptr(),
        .down = down.data_ptr(),
        .dn_scale = dn_scale.data_ptr(),
        .topk_ids = topk_ids.data_ptr(),
        .topk_w = topk_w.data_ptr(),
        .out = out.data_ptr(),
        .M = m, .K = k, .H = h, .I = i,
    };
    LaunchKernel(dim3(m, h / slice_h), threads2, dev,
                 k * i * sizeof(__nv_bfloat16) + 2 * slice_h * i)(moe_g2_kernel<threads2, slice_h>, g2);
  }
};

}  // namespace
