#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace pe115::end_terrain {

// Minimal, observable reimplementation of PE 1.1.5 The End's seed-driven
// terrain.  `generate_base_chunk` exposes the raw terrain source; callers
// that need the fresh central End scene can use `generate_fresh_entry_chunk`.
// Runtime dragon-fight changes and player edits are deliberately not inferred
// from a world seed.
class Generator final {
public:
    static constexpr std::int32_t chunk_width = 16;
    static constexpr std::int32_t chunk_height = 128;
    static constexpr std::size_t block_count =
        static_cast<std::size_t>(chunk_width * chunk_width * chunk_height);

    explicit Generator(std::uint32_t world_seed);
    ~Generator();
    Generator(Generator&&) noexcept;
    Generator& operator=(Generator&&) noexcept;
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;

    // Each byte is either native End stone (121) or air (0). This source
    // writes no other block type before decorator/runtime processing.
    [[nodiscard]] std::array<std::uint8_t, block_count> generate_base_chunk(
        std::int32_t chunk_x,
        std::int32_t chunk_z) const;

    // Reproduces the static, player-visible central scene after first entry:
    // raw terrain, all ten End spikes and their initial crystal support, the
    // inactive exit podium, and the arrival platform.  Outer-island decorator
    // features are handled separately because PE schedules them per chunk.
    [[nodiscard]] std::array<std::uint8_t, block_count>
    generate_fresh_entry_chunk(
        std::int32_t chunk_x,
        std::int32_t chunk_z) const;

    // Adds PE's seed-driven non-city End decorator effects to raw terrain:
    // outer-island features, chorus growth, and natural gateways.  It does
    // not infer runtime dragon-fight state or player edits.
    [[nodiscard]] std::array<std::uint8_t, block_count>
    generate_decorated_chunk(
        std::int32_t chunk_x,
        std::int32_t chunk_z) const;

    // Combines the seed-driven decorator stage with the initial static End
    // scene. End City templates remain a separate structure subsystem.
    [[nodiscard]] std::array<std::uint8_t, block_count>
    generate_fresh_visible_chunk(
        std::int32_t chunk_x,
        std::int32_t chunk_z) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Fixed full-chunk regressions for the embedded implementation. The checked
// chunks cover distinct signed seed bit patterns and terrain regions.
[[nodiscard]] bool self_test() noexcept;

} // namespace pe115::end_terrain
