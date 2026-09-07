#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace pe115::avx2 {

enum class ScanKind {
    all,
    contiguous_range,
    high_word,
    low_word,
};

struct ScanSpec final {
    ScanKind kind;
    std::uint64_t count;
    std::uint64_t range_start;
    std::uint16_t word;
};

// This function is compiled in a separate AVX2 translation unit.  Its caller
// must check this first, so machines without AVX2 remain supported.
[[nodiscard]] bool is_available() noexcept;

// Scan eight independent seeds per vector.  `draws[i]` is the required result
// of PE's (i + 2)-bounded shuffle draw.  Returned seeds are not sorted.
[[nodiscard]] std::vector<std::uint32_t> scan(
    const ScanSpec& specification,
    const std::array<std::uint8_t, 9>& draws);

} // namespace pe115::avx2
