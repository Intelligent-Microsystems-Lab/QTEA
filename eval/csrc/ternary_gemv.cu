/*
 * Fully fused Ternary LUT GEMV — CUDA kernel v2.
 *
 * The B=1 path uses two launches:
 *   1. Precompute per-group half-LUTs and activation sums once
 *   2. Gather packed rows from the precomputed LUTs
 *
 * The batched path keeps the original fused structure:
 *   1. Compute x * v (activation scaling) inline
 *   2. Build LUT in shared memory
 *   3. Gather from packed weights, apply alpha/zero
 *   4. Add sparse FP8 residual contribution
 *
 * Shared memory layout per group:
 *   lut[26][243] floats = ~25 KB (fits in 48 KB shared)
 *   act_sum: 1 float (reduced cooperatively)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define PACKS_PER_GROUP 26
#define LUT_SIZE 243
#define HALF_LUT_SIZE 122
#define GROUP_COLS 128
#define GROUP_CHUNK_SIZE 1
#define CHUNKED_G_THRESHOLD 32
#define TRANSPOSED_G_THRESHOLD 64
#define DIRECT_DECODE_ROW_THRESHOLD 1024
#define HIGHG_UNROLL_THRESHOLD 128

// Fast FP32→FP16 conversion kernel (avoids torch .to() tensor allocation overhead)
__global__ void convert_f32_to_f16(__half* __restrict__ dst, const float* __restrict__ src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// Helper: convert FP32 output to FP16 in-place using pre-allocated buffer
static inline torch::Tensor convert_output_f16(torch::Tensor output_f32, torch::Tensor& f16_buf, int R) {
    if (f16_buf.numel() < R) {
        f16_buf = torch::empty({R}, output_f32.options().dtype(torch::kHalf));
    }
    int blocks = (R + 255) / 256;
    convert_f32_to_f16<<<blocks, 256>>>(
        reinterpret_cast<__half*>(f16_buf.data_ptr<at::Half>()),
        output_f32.data_ptr<float>(), R);
    return f16_buf.slice(0, 0, R).unsqueeze(0);
}

__global__ void ternary_gemv_build_lut_b1_kernel(
    const __half* __restrict__ x,        // (C,)
    const float* __restrict__ col_v,    // (C,)
    __half*      __restrict__ lut_all,  // (G, 26, 122)
    float*       __restrict__ act_sums, // (G,)
    int G, int C
) {
    __shared__ float s_xv[PACKS_PER_GROUP * 5];
    __shared__ float s_reduce[256];

    int g = blockIdx.x;
    if (g >= G) return;

    int c_start = g * GROUP_COLS;

    for (int idx = threadIdx.x; idx < PACKS_PER_GROUP * 5; idx += blockDim.x) {
        int col = c_start + idx;
        s_xv[idx] = (idx < GROUP_COLS && col < C) ? __half2float(x[col]) * col_v[col] : 0.0f;
    }

    float local_sum = 0.0f;
    for (int idx = threadIdx.x; idx < GROUP_COLS; idx += blockDim.x) {
        int col = c_start + idx;
        if (col < C) local_sum += __half2float(x[col]);
    }
    s_reduce[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) act_sums[g] = s_reduce[0];
    __syncthreads();

    __half* lut_g = lut_all + g * PACKS_PER_GROUP * HALF_LUT_SIZE;
    int half_lut = PACKS_PER_GROUP * HALF_LUT_SIZE;
    for (int idx = threadIdx.x; idx < half_lut; idx += blockDim.x) {
        int k = idx / HALF_LUT_SIZE;
        int lut_idx = idx % HALF_LUT_SIZE;
        int xv_base = k * 5;

        int rem = lut_idx;
        float val = 0.0f;
        #pragma unroll
        for (int j = 0; j < 5; j++) {
            int digit = rem % 3;
            rem /= 3;
            val += (float)(digit - 1) * s_xv[xv_base + j];
        }
        lut_g[k * HALF_LUT_SIZE + lut_idx] = __float2half_rn(val);
    }
}


// Fused QKV LUT build: 3 LUTs with shared act_sums.
// Grid: (3*G,), each block builds one (layer, group) LUT.
__global__ void ternary_gemv_build_lut_qkv_b1_kernel(
    const __half* __restrict__ x,           // (C,)
    const float* __restrict__ col_v_q,     // (C,)
    const float* __restrict__ col_v_k,     // (C,)
    const float* __restrict__ col_v_v,     // (C,)
    __half*      __restrict__ lut_all,     // (3, G, 26, 122)
    float*       __restrict__ act_sums,    // (G,) — shared
    int G, int C
) {
    __shared__ float s_xv[PACKS_PER_GROUP * 5];
    __shared__ float s_reduce[256];

    int layer = blockIdx.x / G;   // 0=Q, 1=K, 2=V
    int g = blockIdx.x % G;
    if (g >= G || layer >= 3) return;

    const float* col_v = (layer == 0) ? col_v_q : (layer == 1) ? col_v_k : col_v_v;
    int c_start = g * GROUP_COLS;

    for (int idx = threadIdx.x; idx < PACKS_PER_GROUP * 5; idx += blockDim.x) {
        int col = c_start + idx;
        s_xv[idx] = (idx < GROUP_COLS && col < C) ? __half2float(x[col]) * col_v[col] : 0.0f;
    }

    // Only layer 0 computes act_sums (shared across Q/K/V since x is the same)
    if (layer == 0) {
        float local_sum = 0.0f;
        for (int idx = threadIdx.x; idx < GROUP_COLS; idx += blockDim.x) {
            int col = c_start + idx;
            if (col < C) local_sum += __half2float(x[col]);
        }
        s_reduce[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) act_sums[g] = s_reduce[0];
    }
    __syncthreads();

    __half* lut_g = lut_all + (layer * G + g) * PACKS_PER_GROUP * HALF_LUT_SIZE;
    int half_lut = PACKS_PER_GROUP * HALF_LUT_SIZE;
    for (int idx = threadIdx.x; idx < half_lut; idx += blockDim.x) {
        int k = idx / HALF_LUT_SIZE;
        int lut_idx = idx % HALF_LUT_SIZE;
        int xv_base = k * 5;
        int rem = lut_idx;
        float val = 0.0f;
        #pragma unroll
        for (int j = 0; j < 5; j++) {
            int digit = rem % 3;
            rem /= 3;
            val += (float)(digit - 1) * s_xv[xv_base + j];
        }
        lut_g[k * HALF_LUT_SIZE + lut_idx] = __float2half_rn(val);
    }
}


// Fused QKV gather: one kernel for R_total rows with per-row LUT selection.
__global__ void ternary_gemv_qkv_b1_chunked_kernel(
    const uint8_t* __restrict__ packed_t,   // (G, 26, R_total) uint8
    const __half*  __restrict__ x,          // (C,) raw activations
    const __half*  __restrict__ alpha_t,    // (G, R_total) float16
    const __half*  __restrict__ zero_pt_t,  // (G, R_total) float16
    const int*     __restrict__ sal_cols,
    const float*   __restrict__ sal_values, // (n_sal, R_total) float
    const __half*  __restrict__ lut_all,    // (3, G, 26, 122)
    const float*   __restrict__ act_sums,   // (G,)
    float*         __restrict__ output,     // (R_total,)
    int R_total, int R_q, int R_k, int G, int n_sal, bool add_sparse
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R_total) return;

    int chunk = blockIdx.y;
    int g_start = chunk * GROUP_CHUNK_SIZE;
    int g_end = g_start + GROUP_CHUNK_SIZE;
    if (g_end > G) g_end = G;

    // Select LUT based on row range
    int lut_layer = (r < R_q) ? 0 : (r < R_q + R_k) ? 1 : 2;

    float acc = 0.0f;

    for (int g = g_start; g < g_end; g++) {
        float alpha_g = __half2float(alpha_t[g * R_total + r]);
        float zero_g = __half2float(zero_pt_t[g * R_total + r]);
        const __half* lut_g = lut_all + (lut_layer * G + g) * PACKS_PER_GROUP * HALF_LUT_SIZE;
        const uint8_t* packed_g = packed_t + g * PACKS_PER_GROUP * R_total;

        float lut_sum = 0.0f;
        #pragma unroll 6
        for (int k = 0; k < PACKS_PER_GROUP; k++) {
            int w_byte = (int)packed_g[k * R_total + r];
            const __half* lut_k = lut_g + k * HALF_LUT_SIZE;
            int idx = (w_byte <= HALF_LUT_SIZE - 1) ? w_byte : ((LUT_SIZE - 1) - w_byte);
            float val = __half2float(lut_k[idx]);
            lut_sum += (w_byte <= HALF_LUT_SIZE - 1) ? val : -val;
        }

        acc += alpha_g * lut_sum + zero_g * act_sums[g];
    }

    if (add_sparse && chunk == 0) {
        float sparse_sum = 0.0f;
        for (int s = 0; s < n_sal; s++) {
            int col = sal_cols[s];
            float val = sal_values[s * R_total + r];
            sparse_sum += val * __half2float(x[col]);
        }
        acc += sparse_sum;
    }

    atomicAdd(output + r, acc);
}


// C++ dispatch for fused QKV
torch::Tensor ternary_gemv_fused_qkv(
    torch::Tensor packed_t,         // (G, 26, R_total) uint8
    torch::Tensor col_v_q,          // (C,)
    torch::Tensor col_v_k,          // (C,)
    torch::Tensor col_v_v,          // (C,)
    torch::Tensor alpha_t,          // (G, R_total) half
    torch::Tensor zero_pt_t,        // (G, R_total) half
    torch::Tensor x,                // (1, C) float
    torch::Tensor sal_cols,         // (n_sal,) int32
    torch::Tensor sal_values,       // (n_sal, R_total) float
    torch::Tensor lut_all,          // (3, G, 26, 122) half scratch
    torch::Tensor act_sums,         // (G,) float scratch
    int R_q, int R_k
) {
    // Validate input dtype — fused kernels expect FP16 x
    auto x_half = (x.scalar_type() == torch::kHalf) ? x.contiguous()
                                                      : x.to(torch::kHalf).contiguous();
    int R_total = packed_t.size(2);
    int G = alpha_t.size(0);
    int C = col_v_q.size(0);
    int n_sal = sal_cols.size(0);

    // Build 3 LUTs in one launch
    int lut_threads = 256;
    ternary_gemv_build_lut_qkv_b1_kernel<<<3 * G, lut_threads>>>(
        reinterpret_cast<const __half*>(x_half.data_ptr<at::Half>()),
        col_v_q.data_ptr<float>(),
        col_v_k.data_ptr<float>(),
        col_v_v.data_ptr<float>(),
        reinterpret_cast<__half*>(lut_all.data_ptr<at::Half>()),
        act_sums.data_ptr<float>(),
        G, C
    );

    // Fused gather
    int gather_threads = 128;
    int blocks = (R_total + gather_threads - 1) / gather_threads;
    int chunks = (G + GROUP_CHUNK_SIZE - 1) / GROUP_CHUNK_SIZE;
    dim3 grid(blocks, chunks);

    auto output = torch::empty({R_total}, packed_t.options().dtype(torch::kFloat32));
    cudaMemsetAsync(output.data_ptr<float>(), 0, R_total * sizeof(float));

    ternary_gemv_qkv_b1_chunked_kernel<<<grid, gather_threads>>>(
        packed_t.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(x_half.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(alpha_t.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(zero_pt_t.data_ptr<at::Half>()),
        sal_cols.data_ptr<int>(),
        sal_values.data_ptr<float>(),
        reinterpret_cast<__half*>(lut_all.data_ptr<at::Half>()),
        act_sums.data_ptr<float>(),
        output.data_ptr<float>(),
        R_total, R_q, R_k, G, n_sal, true
    );

    return output.unsqueeze(0);
}


// Fused gate+up: 2 layers with separate col_v, shared act_sums.
// Reuses the QKV LUT build (with col_v_v = col_v_up) and gather kernels.
torch::Tensor ternary_gemv_fused_gateup(
    torch::Tensor packed_t,         // (G, 26, R_total) uint8 — [gate; up] concatenated
    torch::Tensor col_v_gate,       // (C,)
    torch::Tensor col_v_up,         // (C,)
    torch::Tensor alpha_t,          // (G, R_total) half
    torch::Tensor zero_pt_t,        // (G, R_total) half
    torch::Tensor x,                // (1, C) float
    torch::Tensor sal_cols,         // (n_sal,) int32
    torch::Tensor sal_values,       // (n_sal, R_total) float
    torch::Tensor lut_all,          // (2, G, 26, 122) half scratch
    torch::Tensor act_sums,         // (G,) float scratch
    int R_gate
) {
    // Validate input dtype
    auto x_half = (x.scalar_type() == torch::kHalf) ? x.contiguous()
                                                      : x.to(torch::kHalf).contiguous();
    int R_total = packed_t.size(2);
    int G = alpha_t.size(0);
    int C = col_v_gate.size(0);
    int n_sal = sal_cols.size(0);

    // Build 2 LUTs
    int lut_threads = 256;
    ternary_gemv_build_lut_qkv_b1_kernel<<<2 * G, lut_threads>>>(
        reinterpret_cast<const __half*>(x_half.data_ptr<at::Half>()),
        col_v_gate.data_ptr<float>(),
        col_v_up.data_ptr<float>(),
        col_v_up.data_ptr<float>(),  // dummy third, only 2 layers used
        reinterpret_cast<__half*>(lut_all.data_ptr<at::Half>()),
        act_sums.data_ptr<float>(),
        G, C
    );

    int gather_threads = 128;
    int blocks = (R_total + gather_threads - 1) / gather_threads;
    int chunks = (G + GROUP_CHUNK_SIZE - 1) / GROUP_CHUNK_SIZE;
    dim3 grid(blocks, chunks);

    auto output = torch::empty({R_total}, packed_t.options().dtype(torch::kFloat32));
    cudaMemsetAsync(output.data_ptr<float>(), 0, R_total * sizeof(float));

    // R_k = R_up = R_total - R_gate for the row boundary
    int R_up = R_total - R_gate;
    ternary_gemv_qkv_b1_chunked_kernel<<<grid, gather_threads>>>(
        packed_t.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(x_half.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(alpha_t.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(zero_pt_t.data_ptr<at::Half>()),
        sal_cols.data_ptr<int>(),
        sal_values.data_ptr<float>(),
        reinterpret_cast<__half*>(lut_all.data_ptr<at::Half>()),
        act_sums.data_ptr<float>(),
        output.data_ptr<float>(),
        R_total, R_gate, R_up, G, n_sal, true
    );

    return output.unsqueeze(0);
}


__global__ void ternary_gemv_v2_precomputed_b1_kernel(
    const uint8_t* __restrict__ packed_t,   // (G, 26, R) uint8
    const __half*  __restrict__ x,          // (C,) raw activations
    const float*   __restrict__ alpha,      // (R, G) float32
    const float*   __restrict__ zero_pt,    // (R, G) float32
    const int*     __restrict__ sal_cols,
    const float*   __restrict__ sal_values,
    const __half*  __restrict__ lut_all,    // (G, 26, 122)
    const float*   __restrict__ act_sums,   // (G,)
    float*         __restrict__ output,     // (R,) float32
    int R, int G, int n_sal
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;

    float acc = 0.0f;

    for (int g = 0; g < G; g++) {
        float alpha_g = alpha[r * G + g];
        float zero_g = zero_pt[r * G + g];
        const __half* lut_g = lut_all + g * PACKS_PER_GROUP * HALF_LUT_SIZE;
        const uint8_t* packed_g = packed_t + g * PACKS_PER_GROUP * R;

        float lut_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < PACKS_PER_GROUP; k++) {
            int w_byte = (int)packed_g[k * R + r];
            const __half* lut_k = lut_g + k * HALF_LUT_SIZE;
            int idx = (w_byte <= HALF_LUT_SIZE - 1) ? w_byte : ((LUT_SIZE - 1) - w_byte);
            float val = __half2float(lut_k[idx]);
            lut_sum += (w_byte <= HALF_LUT_SIZE - 1) ? val : -val;
        }

        acc += alpha_g * lut_sum + zero_g * act_sums[g];
    }

    float sparse_sum = 0.0f;
    for (int s = 0; s < n_sal; s++) {
        int col = sal_cols[s];
        float val = sal_values[s * R + r];
        sparse_sum += val * __half2float(x[col]);
    }
    output[r] = acc + sparse_sum;
}


__global__ void ternary_gemv_v2_precomputed_b1_chunked_kernel(
    const uint8_t* __restrict__ packed_t,   // (G, 26, R) uint8
    const __half*  __restrict__ x,          // (C,) raw activations
    const __half*  __restrict__ alpha_t,    // (G, R) float16
    const __half*  __restrict__ zero_pt_t,  // (G, R) float16
    const int*     __restrict__ sal_cols,
    const float*   __restrict__ sal_values,
    const __half*  __restrict__ lut_all,    // (G, 26, 122)
    const float*   __restrict__ act_sums,   // (G,)
    float*         __restrict__ output,     // (R,) float32
    int R, int G, int n_sal, bool add_sparse
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;

    int chunk = blockIdx.y;
    int g_start = chunk * GROUP_CHUNK_SIZE;
    int g_end = g_start + GROUP_CHUNK_SIZE;
    if (g_end > G) g_end = G;

    float acc = 0.0f;

    for (int g = g_start; g < g_end; g++) {
        float alpha_g = __half2float(alpha_t[g * R + r]);
        float zero_g = __half2float(zero_pt_t[g * R + r]);
        const __half* lut_g = lut_all + g * PACKS_PER_GROUP * HALF_LUT_SIZE;
        const uint8_t* packed_g = packed_t + g * PACKS_PER_GROUP * R;

        float lut_sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < PACKS_PER_GROUP; k++) {
            int w_byte = (int)packed_g[k * R + r];
            const __half* lut_k = lut_g + k * HALF_LUT_SIZE;
            int idx = (w_byte <= HALF_LUT_SIZE - 1) ? w_byte : ((LUT_SIZE - 1) - w_byte);
            float val = __half2float(lut_k[idx]);
            lut_sum += (w_byte <= HALF_LUT_SIZE - 1) ? val : -val;
        }

        acc += alpha_g * lut_sum + zero_g * act_sums[g];
    }

    if (add_sparse && chunk == 0) {
        float sparse_sum = 0.0f;
        for (int s = 0; s < n_sal; s++) {
            int col = sal_cols[s];
            float val = sal_values[s * R + r];
            sparse_sum += val * __half2float(x[col]);
        }
        acc += sparse_sum;
    }

    atomicAdd(output + r, acc);
}


// High-G variant with partial unroll for better occupancy at G>=128
__global__ void ternary_gemv_v2_precomputed_b1_chunked_highg_kernel(
    const uint8_t* __restrict__ packed_t,   // (G, 26, R) uint8
    const __half*  __restrict__ x,          // (C,) raw activations
    const __half*  __restrict__ alpha_t,    // (G, R) float16
    const __half*  __restrict__ zero_pt_t,  // (G, R) float16
    const int*     __restrict__ sal_cols,
    const float*   __restrict__ sal_values,
    const __half*  __restrict__ lut_all,    // (G, 26, 122)
    const float*   __restrict__ act_sums,   // (G,)
    float*         __restrict__ output,     // (R,) float32
    int R, int G, int n_sal, bool add_sparse
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;

    int chunk = blockIdx.y;
    int g_start = chunk * GROUP_CHUNK_SIZE;
    int g_end = g_start + GROUP_CHUNK_SIZE;
    if (g_end > G) g_end = G;

    float acc = 0.0f;

    for (int g = g_start; g < g_end; g++) {
        float alpha_g = __half2float(alpha_t[g * R + r]);
        float zero_g = __half2float(zero_pt_t[g * R + r]);
        const __half* lut_g = lut_all + g * PACKS_PER_GROUP * HALF_LUT_SIZE;
        const uint8_t* packed_g = packed_t + g * PACKS_PER_GROUP * R;

        float lut_sum = 0.0f;
        #pragma unroll 6
        for (int k = 0; k < PACKS_PER_GROUP; k++) {
            int w_byte = (int)packed_g[k * R + r];
            const __half* lut_k = lut_g + k * HALF_LUT_SIZE;
            int idx = (w_byte <= HALF_LUT_SIZE - 1) ? w_byte : ((LUT_SIZE - 1) - w_byte);
            float val = __half2float(lut_k[idx]);
            lut_sum += (w_byte <= HALF_LUT_SIZE - 1) ? val : -val;
        }

        acc += alpha_g * lut_sum + zero_g * act_sums[g];
    }

    if (add_sparse && chunk == 0) {
        float sparse_sum = 0.0f;
        for (int s = 0; s < n_sal; s++) {
            int col = sal_cols[s];
            float val = sal_values[s * R + r];
            sparse_sum += val * __half2float(x[col]);
        }
        acc += sparse_sum;
    }

    atomicAdd(output + r, acc);
}


__global__ void ternary_gemv_sparse_b1_rowmajor_kernel(
    const int*   __restrict__ sal_cols,
    const __half* __restrict__ sal_values_by_row, // (R, n_sal)
    const __half* __restrict__ x,                 // (C,)
    float*       __restrict__ output,            // (R,)
    int R, int n_sal
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp = tid >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= R) return;

    const __half* vals = sal_values_by_row + warp * n_sal;
    float sum = 0.0f;
    for (int s = lane; s < n_sal; s += 32) {
        sum += __half2float(vals[s]) * __half2float(x[sal_cols[s]]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        output[warp] += sum;
    }
}


__global__ void ternary_gemv_direct_b1_chunked_rowmajor_kernel(
    const uint8_t* __restrict__ packed,     // (R, G * 26) uint8
    const __half*  __restrict__ x,          // (C,) raw activations
    const float*   __restrict__ col_v,      // (C,) per-column scale
    const float*   __restrict__ alpha,      // (R, G) float32
    const float*   __restrict__ zero_pt,    // (R, G) float32
    const int*     __restrict__ sal_cols,
    const float*   __restrict__ sal_values,
    float*         __restrict__ output,     // (R,) float32
    int R, int G, int C, int n_sal
) {
    __shared__ float s_xv[GROUP_COLS];
    __shared__ float s_act_sum;

    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y;
    if (g >= G) return;

    int c_start = g * GROUP_COLS;
    for (int idx = threadIdx.x; idx < GROUP_COLS; idx += blockDim.x) {
        int col = c_start + idx;
        s_xv[idx] = (col < C) ? __half2float(x[col]) * col_v[col] : 0.0f;
    }
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        int c_end = c_start + GROUP_COLS;
        if (c_end > C) c_end = C;
        for (int c = c_start; c < c_end; c++) sum += __half2float(x[c]);
        s_act_sum = sum;
    }
    __syncthreads();

    if (r < R) {
        float lut_sum = 0.0f;
        int packed_stride = G * PACKS_PER_GROUP;
        int packed_base = r * packed_stride + g * PACKS_PER_GROUP;
        #pragma unroll
        for (int k = 0; k < PACKS_PER_GROUP - 1; k++) {
            int rem = (int)packed[packed_base + k];
            int xv_base = k * 5;
            #pragma unroll
            for (int j = 0; j < 5; j++) {
                int digit = rem % 3;
                rem /= 3;
                lut_sum += (float)(digit - 1) * s_xv[xv_base + j];
            }
        }
        int rem = (int)packed[packed_base + PACKS_PER_GROUP - 1];
        #pragma unroll
        for (int j = 0; j < 3; j++) {
            int digit = rem % 3;
            rem /= 3;
            lut_sum += (float)(digit - 1) * s_xv[(PACKS_PER_GROUP - 1) * 5 + j];
        }

        float acc = alpha[r * G + g] * lut_sum + zero_pt[r * G + g] * s_act_sum;
        if (g == 0) {
            float sparse_sum = 0.0f;
            for (int s = 0; s < n_sal; s++) {
                int col = sal_cols[s];
                float val = sal_values[s * R + r];
                sparse_sum += val * __half2float(x[col]);
            }
            acc += sparse_sum;
        }
        atomicAdd(output + r, acc);
    }
}


// Batched version with half-table LUT
__global__ void ternary_gemv_v2_batched_kernel(
    const uint8_t* __restrict__ packed,
    const float*   __restrict__ x,
    const float*   __restrict__ col_v,
    const float*   __restrict__ alpha,
    const float*   __restrict__ zero_pt,
    const int*     __restrict__ sal_cols,
    const float*   __restrict__ sal_values,
    float*         __restrict__ output,
    int B, int R, int G, int C, int n_sal
) {
    __shared__ float lut[PACKS_PER_GROUP][HALF_LUT_SIZE];
    __shared__ float s_act_sum;

    int b = blockIdx.y;
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    const float* x_b = x + b * C;
    float acc = 0.0f;

    for (int g = 0; g < G; g++) {
        int c_start = g * GROUP_COLS;

        int half_lut = PACKS_PER_GROUP * HALF_LUT_SIZE;
        for (int idx = threadIdx.x; idx < half_lut; idx += blockDim.x) {
            int k = idx / HALF_LUT_SIZE;
            int lut_idx = idx % HALF_LUT_SIZE;
            float val = 0.0f;
            int rem = lut_idx;
            int col_base = c_start + k * 5;
            #pragma unroll 5
            for (int j = 0; j < 5; j++) {
                int digit = rem % 3;
                rem /= 3;
                int col = col_base + j;
                if (col < C) {
                    val += (float)(digit - 1) * x_b[col] * col_v[col];
                }
            }
            lut[k][lut_idx] = val;
        }

        if (threadIdx.x == 0) {
            float sum = 0.0f;
            int c_end = c_start + GROUP_COLS;
            if (c_end > C) c_end = C;
            for (int c = c_start; c < c_end; c++) sum += x_b[c];
            s_act_sum = sum;
        }
        __syncthreads();

        if (r < R) {
            float alpha_g = alpha[r * G + g];
            float zero_g = zero_pt[r * G + g];
            float lut_sum = 0.0f;
            int packed_base = r * (G * PACKS_PER_GROUP) + g * PACKS_PER_GROUP;
            #pragma unroll
            for (int k = 0; k < PACKS_PER_GROUP; k++) {
                int w_byte = (int)packed[packed_base + k];
                int idx = (w_byte <= HALF_LUT_SIZE - 1) ? w_byte : ((LUT_SIZE - 1) - w_byte);
                float val = lut[k][idx];
                lut_sum += (w_byte <= HALF_LUT_SIZE - 1) ? val : -val;
            }
            acc += alpha_g * lut_sum + zero_g * s_act_sum;
        }
        __syncthreads();
    }

    if (r < R) {
        float sparse_sum = 0.0f;
        for (int s = 0; s < n_sal; s++) {
            sparse_sum += sal_values[s * R + r] * x_b[sal_cols[s]];
        }
        output[b * R + r] = acc + sparse_sum;
    }
}


torch::Tensor ternary_gemv_fused(
    torch::Tensor packed,           // (R, G*26) uint8 CUDA
    torch::Tensor packed_t,         // (G, 26, R) uint8 CUDA, for B=1 coalesced row loads
    torch::Tensor col_v,            // (C,) float CUDA
    torch::Tensor alpha,            // (R, G) float CUDA
    torch::Tensor zero_pt,          // (R, G) float CUDA
    torch::Tensor alpha_t,          // (G, R) half CUDA for transposed B=1 gather
    torch::Tensor zero_pt_t,        // (G, R) half CUDA for transposed B=1 gather
    torch::Tensor x,                // (B, C) float CUDA
    torch::Tensor sal_cols,         // (n_sal,) int32 CUDA
    torch::Tensor sal_values,       // (n_sal, R) float CUDA — column-major per salient col
    torch::Tensor sal_values_by_row,// (R, n_sal) float CUDA for large B=1 sparse split
    torch::Tensor lut_all,          // (G, 26, 122) half CUDA scratch for B=1
    torch::Tensor act_sums,         // (G,) float CUDA scratch for B=1
    torch::Tensor output_f32_buf,   // (R,) float CUDA scratch — pre-allocated FP32 output
    torch::Tensor output_f16_buf    // (R,) half CUDA scratch for fast FP16 output
) {
    int R = packed.size(0);
    int G = alpha.size(1);
    int C = col_v.size(0);
    int n_sal = sal_cols.size(0);

    // Accept any input dtype. B=1 kernels read FP16 directly (no conversion copy).
    auto input_dtype = x.scalar_type();
    auto x_flat = x.reshape({-1, C});
    auto x_f16 = (input_dtype == torch::kHalf) ? x_flat : x_flat.to(torch::kHalf);
    int B = x_flat.size(0);

    int threads = 256;

    if (B == 1) {
        int gather_threads = 128;
        int blocks = (R + gather_threads - 1) / gather_threads;

        if (G < TRANSPOSED_G_THRESHOLD && R <= DIRECT_DECODE_ROW_THRESHOLD) {
            int direct_threads = (R <= 1024) ? 64 : gather_threads;
            int direct_blocks = (R + direct_threads - 1) / direct_threads;
            // Reuse pre-allocated output buffer (zero it)
            cudaMemsetAsync(output_f32_buf.data_ptr<float>(), 0, R * sizeof(float));
            auto output = output_f32_buf;
            dim3 grid(direct_blocks, G);
            ternary_gemv_direct_b1_chunked_rowmajor_kernel<<<grid, direct_threads>>>(
                packed.data_ptr<uint8_t>(),
                reinterpret_cast<const __half*>(x_f16.data_ptr<at::Half>()),
                col_v.data_ptr<float>(),
                alpha.data_ptr<float>(),
                zero_pt.data_ptr<float>(),
                sal_cols.data_ptr<int>(),
                sal_values.data_ptr<float>(),
                output.data_ptr<float>(),
                R, G, C, n_sal
            );
            if (input_dtype == torch::kHalf) return convert_output_f16(output, output_f16_buf, R);
            return output.unsqueeze(0);
        }

        ternary_gemv_build_lut_b1_kernel<<<G, threads>>>(
            reinterpret_cast<const __half*>(x_f16.data_ptr<at::Half>()),
            col_v.data_ptr<float>(),
            reinterpret_cast<__half*>(lut_all.data_ptr<at::Half>()),
            act_sums.data_ptr<float>(),
            G, C
        );

        if (G >= CHUNKED_G_THRESHOLD) {
            auto output = output_f32_buf;
            cudaMemsetAsync(output.data_ptr<float>(), 0, R * sizeof(float));
            int chunks = (G + GROUP_CHUNK_SIZE - 1) / GROUP_CHUNK_SIZE;
            dim3 grid(blocks, chunks);
            bool split_sparse = (n_sal >= 1024 && sal_values_by_row.size(1) == n_sal);
            if (G >= HIGHG_UNROLL_THRESHOLD) {
                // down (G=200): transposed layout + partial unroll 8
                ternary_gemv_v2_precomputed_b1_chunked_highg_kernel<<<grid, gather_threads>>>(
                    packed_t.data_ptr<uint8_t>(),
                    reinterpret_cast<const __half*>(x_f16.data_ptr<at::Half>()),
                    reinterpret_cast<__half*>(alpha_t.data_ptr<at::Half>()),
                    reinterpret_cast<__half*>(zero_pt_t.data_ptr<at::Half>()),
                    sal_cols.data_ptr<int>(),
                    sal_values.data_ptr<float>(),
                    reinterpret_cast<__half*>(lut_all.data_ptr<at::Half>()),
                    act_sums.data_ptr<float>(),
                    output.data_ptr<float>(),
                    R, G, n_sal, !split_sparse
                );
            } else if (G >= TRANSPOSED_G_THRESHOLD) {
                // O (G=64): transposed layout + full unroll
                ternary_gemv_v2_precomputed_b1_chunked_kernel<<<grid, gather_threads>>>(
                    packed_t.data_ptr<uint8_t>(),
                    reinterpret_cast<const __half*>(x_f16.data_ptr<at::Half>()),
                    reinterpret_cast<__half*>(alpha_t.data_ptr<at::Half>()),
                    reinterpret_cast<__half*>(zero_pt_t.data_ptr<at::Half>()),
                    sal_cols.data_ptr<int>(),
                    sal_values.data_ptr<float>(),
                    reinterpret_cast<__half*>(lut_all.data_ptr<at::Half>()),
                    act_sums.data_ptr<float>(),
                    output.data_ptr<float>(),
                    R, G, n_sal, !split_sparse
                );
            } else {
                // gate/up (G=40): transposed layout + partial unroll 8
                ternary_gemv_v2_precomputed_b1_chunked_highg_kernel<<<grid, gather_threads>>>(
                    packed_t.data_ptr<uint8_t>(),
                    reinterpret_cast<const __half*>(x_f16.data_ptr<at::Half>()),
                    reinterpret_cast<__half*>(alpha_t.data_ptr<at::Half>()),
                    reinterpret_cast<__half*>(zero_pt_t.data_ptr<at::Half>()),
                    sal_cols.data_ptr<int>(),
                    sal_values.data_ptr<float>(),
                    reinterpret_cast<__half*>(lut_all.data_ptr<at::Half>()),
                    act_sums.data_ptr<float>(),
                    output.data_ptr<float>(),
                    R, G, n_sal, !split_sparse
                );
            }
            if (split_sparse) {
                int sparse_threads = 256;
                int warps_per_block = sparse_threads / 32;
                int sparse_blocks = (R + warps_per_block - 1) / warps_per_block;
                ternary_gemv_sparse_b1_rowmajor_kernel<<<sparse_blocks, sparse_threads>>>(
                    sal_cols.data_ptr<int>(),
                    reinterpret_cast<__half*>(sal_values_by_row.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(x_f16.data_ptr<at::Half>()),
                    output.data_ptr<float>(),
                    R, n_sal
                );
            }
            if (input_dtype == torch::kHalf) return convert_output_f16(output, output_f16_buf, R);
            return output.unsqueeze(0);
        } else {
            auto output = torch::empty({R}, packed.options().dtype(torch::kFloat32));
            ternary_gemv_v2_precomputed_b1_kernel<<<blocks, gather_threads>>>(
                packed_t.data_ptr<uint8_t>(),
                reinterpret_cast<const __half*>(x_f16.data_ptr<at::Half>()),
                alpha.data_ptr<float>(),
                zero_pt.data_ptr<float>(),
                sal_cols.data_ptr<int>(),
                sal_values.data_ptr<float>(),
                reinterpret_cast<__half*>(lut_all.data_ptr<at::Half>()),
                act_sums.data_ptr<float>(),
                output.data_ptr<float>(),
                R, G, n_sal
            );
            if (input_dtype == torch::kHalf) return convert_output_f16(output, output_f16_buf, R);
            return output.unsqueeze(0);
        }
    } else {
        auto x_f32 = x_flat.to(torch::kFloat32);
        auto output = torch::empty({B, R}, packed.options().dtype(torch::kFloat32));
        dim3 grid((R + threads - 1) / threads, B);

        ternary_gemv_v2_batched_kernel<<<grid, threads>>>(
            packed.data_ptr<uint8_t>(),
            x_f32.data_ptr<float>(),
            col_v.data_ptr<float>(),
            alpha.data_ptr<float>(),
            zero_pt.data_ptr<float>(),
            sal_cols.data_ptr<int>(),
            sal_values.data_ptr<float>(),
            output.data_ptr<float>(),
            B, R, G, C, n_sal
        );
        return (input_dtype == torch::kHalf) ? output.to(torch::kHalf) : output;
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ternary_gemv_fused", &ternary_gemv_fused, "Fused Ternary LUT GEMV v2 (CUDA)");
    m.def("ternary_gemv_fused_qkv", &ternary_gemv_fused_qkv, "Fused QKV Ternary LUT GEMV (CUDA)");
    m.def("ternary_gemv_fused_gateup", &ternary_gemv_fused_gateup, "Fused Gate+Up Ternary LUT GEMV (CUDA)");
}
