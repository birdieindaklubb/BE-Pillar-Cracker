// Optional CUDA implementation of the exact PE 1.1.5 pillar predicate.
// Every GPU thread evaluates a whole candidate seed; this is intentionally
// simple data parallelism rather than a different or approximate RNG model.

#include "cuda_backend.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace pe115::cuda_backend {
namespace {

constexpr std::uint32_t kMtMultiplier = 1'812'433'253U;
constexpr std::uint32_t kMtUpperMask = 0x8000'0000U;
constexpr std::uint32_t kMtLowerMask = 0x7fff'ffffU;
constexpr std::uint32_t kMtMatrixA = 0x9908'b0dfU;
constexpr std::uint32_t kCandidateCapacity = 65'536U;
constexpr std::uint64_t kMaximumLaunchBatch = 1ULL << 26;
constexpr unsigned int kThreadsPerBlock = 512U;
constexpr unsigned int kResidentGridMultiplier = 4U;

__constant__ std::uint8_t kDraws[9];
__constant__ std::uint16_t kAllowedShapes[10];

[[nodiscard]] __device__ std::uint32_t temper(std::uint32_t value) {
    value ^= value >> 11U;
    value ^= (value << 7U) & 0x9d2c'5680U;
    value ^= (value << 15U) & 0xefc6'0000U;
    value ^= value >> 18U;
    return value;
}

[[nodiscard]] __device__ std::uint32_t seed_step(
    std::uint32_t value,
    std::uint32_t index) {
    return kMtMultiplier * (value ^ (value >> 30U)) + index;
}

[[nodiscard]] __device__ __forceinline__ bool residue_matches(
    std::uint32_t value,
    std::uint32_t draw_index,
    std::uint8_t expected) {
    // Fixed divisors let NVCC choose its constant-reduction sequence rather
    // than emit a general runtime division.
    switch (draw_index) {
    case 0U:
        return (value & 1U) == expected;
    case 1U:
        return value % 3U == expected;
    case 2U:
        return (value & 3U) == expected;
    case 3U:
        return value % 5U == expected;
    case 4U:
        return value % 6U == expected;
    case 5U:
        return value % 7U == expected;
    case 6U:
        return (value & 7U) == expected;
    case 7U:
        return value % 9U == expected;
    case 8U:
        return value % 10U == expected;
    default:
        return false;
    }
}

[[nodiscard]] __device__ std::uint32_t seed_at(
    int scan_kind,
    std::uint64_t range_start,
    std::uint16_t word,
    std::uint64_t index) {
    switch (scan_kind) {
    case 0: // all
        return static_cast<std::uint32_t>(index);
    case 1: // contiguous range
        return static_cast<std::uint32_t>(range_start + index);
    case 2: // high word
        return (static_cast<std::uint32_t>(word) << 16U)
            | static_cast<std::uint32_t>(index);
    case 3: // low word
        return (static_cast<std::uint32_t>(index) << 16U) | word;
    default:
        return 0U;
    }
}

__global__ void scan_kernel(
    int scan_kind,
    std::uint64_t range_start,
    std::uint16_t word,
    std::uint64_t index_base,
    std::uint64_t count,
    std::uint32_t* candidates,
    std::uint32_t* candidate_count) {
    const std::uint64_t first = static_cast<std::uint64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x)
        * blockDim.x;

    for (std::uint64_t local_index = first; local_index < count;
         local_index += stride) {
        const std::uint32_t seed = seed_at(
            scan_kind, range_start, word, index_base + local_index);
        std::uint32_t initial_low[10];
        initial_low[0] = seed;
        std::uint32_t state = seed;
        for (std::uint32_t index = 1; index <= 397; ++index) {
            state = seed_step(state, index);
            if (index < 10U) {
                initial_low[index] = state;
            }
        }

        bool matches = true;
#pragma unroll
        for (std::uint32_t index = 0; index < 9U; ++index) {
            const std::uint32_t joined =
                (initial_low[index] & kMtUpperMask)
                | (initial_low[index + 1U] & kMtLowerMask);
            const std::uint32_t twisted = state ^ (joined >> 1U)
                ^ ((0U - (joined & 1U)) & kMtMatrixA);
            if (!residue_matches(temper(twisted), index, kDraws[index])) {
                matches = false;
                break;
            }
            if (index != 8U) {
                state = seed_step(state, 398U + index);
            }
        }

        if (matches) {
            const std::uint32_t slot = atomicAdd(candidate_count, 1U);
            if (slot < kCandidateCapacity) {
                candidates[slot] = seed;
            }
        }
    }
}

__global__ void scan_mask_kernel(
    int scan_kind,
    std::uint64_t range_start,
    std::uint16_t word,
    std::uint64_t index_base,
    std::uint64_t count,
    std::uint32_t* candidates,
    std::uint32_t* candidate_count) {
    const std::uint64_t first = static_cast<std::uint64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x)
        * blockDim.x;

    for (std::uint64_t local_index = first; local_index < count;
         local_index += stride) {
        const std::uint32_t seed = seed_at(
            scan_kind, range_start, word, index_base + local_index);
        std::uint32_t initial_low[10];
        initial_low[0] = seed;
        std::uint32_t state = seed;
        for (std::uint32_t index = 1; index <= 397U; ++index) {
            state = seed_step(state, index);
            if (index < 10U) {
                initial_low[index] = state;
            }
        }

        std::uint32_t random_values[9];
#pragma unroll
        for (std::uint32_t index = 0; index < 9U; ++index) {
            const std::uint32_t joined =
                (initial_low[index] & kMtUpperMask)
                | (initial_low[index + 1U] & kMtLowerMask);
            const std::uint32_t twisted = state ^ (joined >> 1U)
                ^ ((0U - (joined & 1U)) & kMtMatrixA);
            random_values[index] = temper(twisted);
            if (index != 8U) {
                state = seed_step(state, 398U + index);
            }
        }

        std::uint8_t order[10];
#pragma unroll
        for (std::uint8_t index = 0U; index < 10U; ++index) {
            order[index] = index;
        }
#pragma unroll
        for (std::uint32_t index = 1U; index < 10U; ++index) {
            const std::uint32_t selected = random_values[index - 1U]
                % (index + 1U);
            const std::uint8_t replacement = order[selected];
            order[selected] = order[index];
            order[index] = replacement;
        }

        bool matches = true;
#pragma unroll
        for (std::uint32_t index = 0U; index < 10U; ++index) {
            const std::uint16_t shape_bit = static_cast<std::uint16_t>(
                1U << order[index]);
            if ((kAllowedShapes[index] & shape_bit) == 0U) {
                matches = false;
                break;
            }
        }
        if (matches) {
            const std::uint32_t slot = atomicAdd(candidate_count, 1U);
            if (slot < kCandidateCapacity) {
                candidates[slot] = seed;
            }
        }
    }
}

void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": "
            + cudaGetErrorString(result));
    }
}

[[nodiscard]] int scan_kind_value(avx2::ScanKind kind) noexcept {
    switch (kind) {
    case avx2::ScanKind::all:
        return 0;
    case avx2::ScanKind::contiguous_range:
        return 1;
    case avx2::ScanKind::high_word:
        return 2;
    case avx2::ScanKind::low_word:
        return 3;
    }
    return 0;
}

} // namespace

bool is_available() noexcept {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

std::vector<std::uint32_t> scan(
    const avx2::ScanSpec& specification,
    const std::array<std::uint8_t, 9>& draws) {
    check(cudaSetDevice(0), "selecting CUDA device 0");
    check(cudaMemcpyToSymbol(kDraws, draws.data(), draws.size()),
        "uploading shuffle constraints");

    cudaDeviceProp device{};
    check(cudaGetDeviceProperties(&device, 0), "querying CUDA device");
    int active_blocks_per_sm = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm, scan_kernel, kThreadsPerBlock, 0),
        "querying CUDA kernel occupancy");
    const std::uint64_t resident_grid_blocks = std::max(
        1ULL, static_cast<std::uint64_t>(active_blocks_per_sm)
            * static_cast<std::uint64_t>(device.multiProcessorCount)
            * kResidentGridMultiplier);

    std::uint32_t* device_candidates = nullptr;
    std::uint32_t* device_count = nullptr;
    try {
        check(cudaMalloc(&device_candidates,
            kCandidateCapacity * sizeof(*device_candidates)),
            "allocating CUDA candidate buffer");
        check(cudaMalloc(&device_count, sizeof(*device_count)),
            "allocating CUDA candidate counter");

        std::vector<std::uint32_t> result;
        const int kind = scan_kind_value(specification.kind);

        const auto collect_segment = [&](auto&& self,
                                         std::uint64_t base,
                                         std::uint64_t count) -> void {
            check(cudaMemset(device_count, 0, sizeof(*device_count)),
                "clearing CUDA candidate counter");
            const std::uint64_t requested_blocks =
                (count + kThreadsPerBlock - 1U) / kThreadsPerBlock;
            const auto blocks = static_cast<unsigned int>(std::min(
                requested_blocks, resident_grid_blocks));
            scan_kernel<<<blocks, kThreadsPerBlock>>>(
                kind, specification.range_start, specification.word, base, count,
                device_candidates, device_count);
            check(cudaGetLastError(), "launching CUDA pillar scan");
            check(cudaDeviceSynchronize(), "running CUDA pillar scan");

            std::uint32_t found = 0;
            check(cudaMemcpy(&found, device_count, sizeof(found),
                cudaMemcpyDeviceToHost), "reading CUDA candidate counter");
            if (found > kCandidateCapacity) {
                if (count <= 1U) {
                    throw std::runtime_error("CUDA candidate buffer overflow "
                        "for a single seed.");
                }
                const std::uint64_t left_count = count / 2U;
                self(self, base, left_count);
                self(self, base + left_count, count - left_count);
                return;
            }

            std::vector<std::uint32_t> batch(found);
            if (found != 0U) {
                check(cudaMemcpy(batch.data(), device_candidates,
                    found * sizeof(*device_candidates), cudaMemcpyDeviceToHost),
                    "reading CUDA candidates");
                result.insert(result.end(), batch.begin(), batch.end());
            }
        };

        for (std::uint64_t base = 0; base < specification.count;
             base += kMaximumLaunchBatch) {
            collect_segment(collect_segment, base, std::min(
                kMaximumLaunchBatch, specification.count - base));
        }

        check(cudaFree(device_count), "freeing CUDA candidate counter");
        device_count = nullptr;
        check(cudaFree(device_candidates), "freeing CUDA candidate buffer");
        device_candidates = nullptr;
        return result;
    } catch (...) {
        if (device_count != nullptr) {
            cudaFree(device_count);
        }
        if (device_candidates != nullptr) {
            cudaFree(device_candidates);
        }
        throw;
    }
}

std::vector<std::uint32_t> scan(
    const avx2::ScanSpec& specification,
    const PillarShapeMasks& allowed_shapes) {
    check(cudaSetDevice(0), "selecting CUDA device 0");
    check(cudaMemcpyToSymbol(kAllowedShapes, allowed_shapes.data(),
        allowed_shapes.size() * sizeof(allowed_shapes.front())),
        "uploading pillar shape constraints");

    cudaDeviceProp device{};
    check(cudaGetDeviceProperties(&device, 0), "querying CUDA device");
    int active_blocks_per_sm = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm, scan_mask_kernel, kThreadsPerBlock, 0),
        "querying CUDA generic-kernel occupancy");
    const std::uint64_t resident_grid_blocks = std::max(
        1ULL, static_cast<std::uint64_t>(active_blocks_per_sm)
            * static_cast<std::uint64_t>(device.multiProcessorCount)
            * kResidentGridMultiplier);

    std::uint32_t* device_candidates = nullptr;
    std::uint32_t* device_count = nullptr;
    try {
        check(cudaMalloc(&device_candidates,
            kCandidateCapacity * sizeof(*device_candidates)),
            "allocating CUDA candidate buffer");
        check(cudaMalloc(&device_count, sizeof(*device_count)),
            "allocating CUDA candidate counter");

        std::vector<std::uint32_t> result;
        const int kind = scan_kind_value(specification.kind);

        const auto collect_segment = [&](auto&& self,
                                         std::uint64_t base,
                                         std::uint64_t count) -> void {
            check(cudaMemset(device_count, 0, sizeof(*device_count)),
                "clearing CUDA candidate counter");
            const std::uint64_t requested_blocks =
                (count + kThreadsPerBlock - 1U) / kThreadsPerBlock;
            const auto blocks = static_cast<unsigned int>(std::min(
                requested_blocks, resident_grid_blocks));
            scan_mask_kernel<<<blocks, kThreadsPerBlock>>>(
                kind, specification.range_start, specification.word, base, count,
                device_candidates, device_count);
            check(cudaGetLastError(), "launching CUDA generic pillar scan");
            check(cudaDeviceSynchronize(), "running CUDA generic pillar scan");

            std::uint32_t found = 0;
            check(cudaMemcpy(&found, device_count, sizeof(found),
                cudaMemcpyDeviceToHost), "reading CUDA candidate counter");
            if (found > kCandidateCapacity) {
                if (count <= 1U) {
                    throw std::runtime_error("CUDA candidate buffer overflow "
                        "for a single seed.");
                }
                const std::uint64_t left_count = count / 2U;
                self(self, base, left_count);
                self(self, base + left_count, count - left_count);
                return;
            }

            std::vector<std::uint32_t> batch(found);
            if (found != 0U) {
                check(cudaMemcpy(batch.data(), device_candidates,
                    found * sizeof(*device_candidates), cudaMemcpyDeviceToHost),
                    "reading CUDA candidates");
                result.insert(result.end(), batch.begin(), batch.end());
            }
        };

        for (std::uint64_t base = 0; base < specification.count;
             base += kMaximumLaunchBatch) {
            collect_segment(collect_segment, base, std::min(
                kMaximumLaunchBatch, specification.count - base));
        }

        check(cudaFree(device_count), "freeing CUDA candidate counter");
        device_count = nullptr;
        check(cudaFree(device_candidates), "freeing CUDA candidate buffer");
        device_candidates = nullptr;
        return result;
    } catch (...) {
        if (device_count != nullptr) {
            cudaFree(device_count);
        }
        if (device_candidates != nullptr) {
            cudaFree(device_candidates);
        }
        throw;
    }
}

} // namespace pe115::cuda_backend
