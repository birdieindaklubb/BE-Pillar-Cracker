#pragma once

#include "pillar_constraints.hpp"

#include <array>
#include <cstdint>

namespace pe115::pillars {

// Native End-spike ring order.  A shape is shuffled onto each site using the
// world-seeded MT stream; the shape determines its height, radius, and cage.
struct Site final {
    std::int32_t x{};
    std::int32_t z{};
};

constexpr std::array<Site, kPillarCount> sites{{
    {42, 0}, {33, 24}, {12, 39}, {-12, 39}, {-33, 24},
    {-42, 0}, {-33, -24}, {-12, -39}, {12, -39}, {33, -24},
}};

constexpr std::uint32_t mt_multiplier = 1'812'433'253U;
constexpr std::uint32_t mt_upper_mask = 0x8000'0000U;
constexpr std::uint32_t mt_lower_mask = 0x7fff'ffffU;
constexpr std::uint32_t mt_matrix_a = 0x9908'b0dfU;

[[nodiscard]] constexpr std::uint32_t seed_step(
    std::uint32_t previous,
    std::uint32_t index) noexcept {
    return mt_multiplier * (previous ^ (previous >> 30U)) + index;
}

[[nodiscard]] constexpr std::uint32_t temper(std::uint32_t value) noexcept {
    value ^= value >> 11U;
    value ^= (value << 7U) & 0x9d2c'5680U;
    value ^= (value << 15U) & 0xefc6'0000U;
    return value ^ (value >> 18U);
}

[[nodiscard]] constexpr std::uint8_t radius_for_shape(
    std::uint8_t shape) noexcept {
    return static_cast<std::uint8_t>(shape / 3U + 2U);
}

[[nodiscard]] constexpr bool is_caged_shape(std::uint8_t shape) noexcept {
    return shape == 1U || shape == 2U;
}

[[nodiscard]] constexpr std::int32_t feature_height_for_shape(
    std::uint8_t shape) noexcept {
    return 76 + static_cast<std::int32_t>(shape) * 3;
}

// The nine MT outputs consumed by PE's forward Fisher-Yates End-spike
// shuffle.  This avoids generating unrelated MT outputs while remaining
// exactly equivalent to the full first twist.
[[nodiscard]] std::array<std::uint32_t, kPillarCount - 1U> shuffle_words(
    std::uint32_t world_seed) noexcept;

[[nodiscard]] std::array<std::uint8_t, kPillarCount> shapes(
    std::uint32_t world_seed) noexcept;

} // namespace pe115::pillars
