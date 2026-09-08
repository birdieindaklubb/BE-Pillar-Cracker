#pragma once

#include "avx2.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace pe115::cuda_backend {

// The CUDA translation unit is optional at build time.  These functions are
// called only when it is present and an NVIDIA CUDA device is usable.
[[nodiscard]] bool is_available() noexcept;

// Exact GPU search of the same seed predicate as the scalar and AVX2 paths.
// Returned seeds are not sorted.
[[nodiscard]] std::vector<std::uint32_t> scan(
    const avx2::ScanSpec& specification,
    const std::array<std::uint8_t, 9>& draws);

// Exact general-layout path for partial height, radius, and cage constraints.
[[nodiscard]] std::vector<std::uint32_t> scan(
    const avx2::ScanSpec& specification,
    const PillarShapeMasks& allowed_shapes);

} // namespace pe115::cuda_backend
