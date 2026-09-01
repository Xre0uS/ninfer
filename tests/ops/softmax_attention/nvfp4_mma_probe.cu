#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::test {
namespace {

__device__ __forceinline__ int probe_swz(int row, int logical_byte) {
    return ((((logical_byte >> 4) ^ (row & 7)) << 4) | (logical_byte & 15));
}

__global__ void nvfp4_mma_probe_kernel(const std::uint8_t* a, const std::uint8_t* b,
                                       const std::uint8_t* a_scale, const std::uint8_t* b_scale,
                                       float* out) {
    __shared__ __align__(16) std::uint8_t a_s[16 * 128];
    __shared__ __align__(16) std::uint8_t b_s[8 * 128];
    const int lane = static_cast<int>(threadIdx.x);
    for (int task = lane; task < 16 * 8; task += 32) {
        const int row     = task / 8;
        const int segment = task - row * 8;
        ops::store_vec(a_s + row * 128 + probe_swz(row, segment * 16),
                       ops::load_vec<int4>(a + row * 128 + segment * 16));
    }
    for (int task = lane; task < 8 * 8; task += 32) {
        const int row     = task / 8;
        const int segment = task - row * 8;
        ops::store_vec(b_s + row * 128 + probe_swz(row, segment * 16),
                       ops::load_vec<int4>(b + row * 128 + segment * 16));
    }
    __syncthreads();

    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) << 4;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) << 4;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row       = lane >> 2;
    float c0 = 0.0F, c1 = 0.0F, c2 = 0.0F, c3 = 0.0F;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        unsigned af[4];
        unsigned bf[2];
        const int abyte = kk * 32 + a_column_byte;
        const int bbyte = kk * 32 + b_column_byte;
        ops::ldmatrix_x4(af[0], af[1], af[2], af[3],
                         ops::smem_addr(a_s + a_row_offset * 128 + probe_swz(a_row_offset, abyte)));
        ops::ldmatrix_x2(bf[0], bf[1],
                         ops::smem_addr(b_s + b_row_offset * 128 + probe_swz(b_row_offset, bbyte)));
        const unsigned as = ops::load_vec<unsigned>(a_scale + sfa_row * 16 + kk * 4);
        const unsigned bs = ops::load_vec<unsigned>(b_scale + sfb_row * 16 + kk * 4);
        ops::mma_nvfp4_e4m3(c0, c1, c2, c3, af[0], af[1], af[2], af[3], bf[0], bf[1], as, bs);
    }
    const int gid                 = lane >> 2;
    const int lid                 = lane & 3;
    const int col0                = 2 * lid;
    out[gid * 8 + col0]           = c0;
    out[gid * 8 + col0 + 1]       = c1;
    out[(gid + 8) * 8 + col0]     = c2;
    out[(gid + 8) * 8 + col0 + 1] = c3;
}

} // namespace

void launch_nvfp4_mma_probe(const std::uint8_t* a, const std::uint8_t* b,
                            const std::uint8_t* a_scale, const std::uint8_t* b_scale, float* out,
                            cudaStream_t stream) {
    nvfp4_mma_probe_kernel<<<1, 32, 0, stream>>>(a, b, a_scale, b_scale, out);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::test
