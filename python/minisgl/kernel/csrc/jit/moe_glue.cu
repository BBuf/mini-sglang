#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace {

// Decode-shape MoE glue around the triton GEMMs:
// - tiny_align: single-block replacement for sgl moe_align_block_size when
//   topk_ids.numel() <= kThreads (bs<=16 decode): counting sort + block pad.
// - sum_add: out[t] = sum_s w[t,s] * c3[t,s,:] + shared[t,:], replacing
//   moe_sum_reduce plus the separate shared-expert add (gemm2 runs with
//   mul_routed_weight=False under no_combine).

struct TinyAlignParams {
  const void *__restrict__ topk_ids; // [M, K] int32
  void *__restrict__ sorted;         // [numel * bm] int32
  void *__restrict__ expert_blocks;  // [numel] int32
  void *__restrict__ total;          // [1] int32
  std::size_t n;                     // numel = M * K
  std::size_t num_experts;
  std::size_t bm;                    // BLOCK_SIZE_M of the gemm1 config
};

template <std::size_t kThreads>
__global__ __launch_bounds__(kThreads) void tiny_align_kernel(
    const __grid_constant__ TinyAlignParams p) {
  extern __shared__ std::int32_t sm[];
  auto *cnt = sm;                    // [E] raw counts -> padded counts
  auto *scan = sm + p.num_experts;   // [E] inclusive scan of padded counts
  auto *cur = scan + p.num_experts;  // [E] scatter cursors
  auto *sids = cur + p.num_experts;  // [n] cached expert of each flat slot

  const auto tid = threadIdx.x;
  if (tid < p.num_experts) cnt[tid] = 0;
  __syncthreads();
  if (tid < p.n) {
    sids[tid] = static_cast<const std::int32_t *>(p.topk_ids)[tid];
    atomicAdd(&cnt[sids[tid]], 1);
  }
  __syncthreads();
  // pad each expert's count to a multiple of bm
  std::int32_t padded = 0;
  if (tid < p.num_experts) {
    padded = static_cast<std::int32_t>((cnt[tid] + p.bm - 1) / p.bm * p.bm);
    scan[tid] = padded;
  }
  __syncthreads();
  // Hillis-Steele inclusive scan over num_experts (<= kThreads) entries
  for (std::size_t d = 1; d < p.num_experts; d <<= 1) {
    std::int32_t v = 0;
    if (tid < p.num_experts && tid >= d) v = scan[tid - d];
    __syncthreads();
    if (tid < p.num_experts) scan[tid] += v;
    __syncthreads();
  }
  const auto total = scan[p.num_experts - 1];
  if (tid == 0) static_cast<std::int32_t *>(p.total)[0] = total;
  // deterministically fill BOTH outputs end to end (hot server memory means
  // torch.empty tails are real garbage; the triton kernels must never see it),
  // then scatter real entries over the pad value
  for (std::size_t i = tid; i < p.n * p.bm; i += kThreads)
    static_cast<std::int32_t *>(p.sorted)[i] = static_cast<std::int32_t>(p.n);
  for (std::size_t i = tid; i < p.n; i += kThreads)
    static_cast<std::int32_t *>(p.expert_blocks)[i] = 0;
  __syncthreads();
  if (tid < p.num_experts) {
    cur[tid] = 0;
    const auto off = scan[tid] - padded;
    for (std::int32_t b = 0; b < padded / static_cast<std::int32_t>(p.bm); ++b)
      static_cast<std::int32_t *>(p.expert_blocks)[off / p.bm + b] =
          static_cast<std::int32_t>(tid);
  }
  __syncthreads();
  if (tid < p.n) {
    const auto e = sids[tid];
    const auto slot = atomicAdd(&cur[e], 1);
    const auto off = scan[e] - static_cast<std::int32_t>((cnt[e] + p.bm - 1) / p.bm * p.bm);
    static_cast<std::int32_t *>(p.sorted)[off + slot] =
        static_cast<std::int32_t>(tid);
  }
}

struct SumAddParams {
  const void *__restrict__ c3;     // [M, K, H] bf16 (unweighted down outputs)
  const void *__restrict__ topk_w; // [M, K] float32
  const void *__restrict__ shared; // [M, H] bf16
  void *__restrict__ out;          // [M, H] bf16
  std::size_t K, H;
};

template <std::size_t kThreads, std::size_t kVec>
__global__ __launch_bounds__(kThreads) void sum_add_kernel(
    const __grid_constant__ SumAddParams p) {
  const auto t = blockIdx.x;
  const auto h = (blockIdx.y * kThreads + threadIdx.x) * kVec;
  if (h >= p.H) return;

  float w[16];
  for (std::size_t s = 0; s < p.K; ++s)
    w[s] = static_cast<const float *>(p.topk_w)[t * p.K + s];

  float acc[kVec];
  const auto *sh = static_cast<const __nv_bfloat16 *>(p.shared) + t * p.H + h;
#pragma unroll
  for (std::size_t v = 0; v < kVec; ++v) acc[v] = __bfloat162float(sh[v]);
  for (std::size_t s = 0; s < p.K; ++s) {
    const auto *c =
        static_cast<const __nv_bfloat16 *>(p.c3) + (t * p.K + s) * p.H + h;
#pragma unroll
    for (std::size_t v = 0; v < kVec; ++v)
      acc[v] += w[s] * __bfloat162float(c[v]);
  }
  auto *o = static_cast<__nv_bfloat16 *>(p.out) + t * p.H + h;
#pragma unroll
  for (std::size_t v = 0; v < kVec; ++v) o[v] = __float2bfloat16(acc[v]);
}

template <std::size_t threads = 256>
struct MoeGlueKernel {
  static void align(const tvm::ffi::TensorView topk_ids, const tvm::ffi::TensorView sorted,
                    const tvm::ffi::TensorView expert_blocks, const tvm::ffi::TensorView total,
                    int64_t num_experts, int64_t bm) {
    using namespace host;
    auto M = SymbolicSize{"M"};
    auto K = SymbolicSize{"K"};
    auto device_ = SymbolicDevice{};
    auto i32 = SymbolicDType{};
    TensorMatcher({M, K}).with_device<kDLCUDA>(device_).with_dtype(i32).verify(topk_ids);
    const auto n = static_cast<std::size_t>(M.unwrap() * K.unwrap());
    RuntimeCheck(n <= threads);
    RuntimeCheck(static_cast<std::size_t>(num_experts) <= threads);
    const auto i32_dt = i32.unwrap();
    RuntimeCheck(i32_dt.code == DLDataTypeCode::kDLInt && i32_dt.bits == 32);

    const auto p = TinyAlignParams{
        .topk_ids = topk_ids.data_ptr(),
        .sorted = sorted.data_ptr(),
        .expert_blocks = expert_blocks.data_ptr(),
        .total = total.data_ptr(),
        .n = n,
        .num_experts = static_cast<std::size_t>(num_experts),
        .bm = static_cast<std::size_t>(bm),
    };
    const auto smem = (3 * num_experts + n) * sizeof(std::int32_t);
    LaunchKernel(1, threads, device_.unwrap(), smem)(tiny_align_kernel<threads>, p);
  }

  static void sum_add(const tvm::ffi::TensorView c3, const tvm::ffi::TensorView topk_w,
                      const tvm::ffi::TensorView shared, const tvm::ffi::TensorView out) {
    using namespace host;
    auto M = SymbolicSize{"M"};
    auto K = SymbolicSize{"K"};
    auto H = SymbolicSize{"H"};
    auto device_ = SymbolicDevice{};
    auto bf = SymbolicDType{};
    auto f32 = SymbolicDType{};
    TensorMatcher({M, K, H}).with_device<kDLCUDA>(device_).with_dtype(bf).verify(c3);
    TensorMatcher({M, K}).with_device<kDLCUDA>(device_).with_dtype(f32).verify(topk_w);
    TensorMatcher({M, H}).with_device<kDLCUDA>(device_).with_dtype(bf).verify(shared);
    TensorMatcher({M, H}).with_device<kDLCUDA>(device_).with_dtype(bf).verify(out);
    const auto bf_dt = bf.unwrap();
    RuntimeCheck(bf_dt.code == DLDataTypeCode::kDLBfloat && bf_dt.bits == 16);
    RuntimeCheck(K.unwrap() <= 16);

    constexpr std::size_t kVec = 4;
    const auto h = static_cast<std::size_t>(H.unwrap());
    RuntimeCheck(h % (kVec * 64) == 0);
    const auto p = SumAddParams{
        .c3 = c3.data_ptr(),
        .topk_w = topk_w.data_ptr(),
        .shared = shared.data_ptr(),
        .out = out.data_ptr(),
        .K = static_cast<std::size_t>(K.unwrap()),
        .H = h,
    };
    const auto per_block = 256 * kVec;
    LaunchKernel(dim3(M.unwrap(), (h + per_block - 1) / per_block), 256,
                 device_.unwrap())(sum_add_kernel<256, kVec>, p);
  }
};

}  // namespace
