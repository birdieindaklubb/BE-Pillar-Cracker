#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace pe115::end_terrain {

// Minimal, observable reimplementation of the PE 1.1.5 The End base terrain
// source.  Its output is deliberately limited to natural terrain: before End
// spikes, the exit podium, the arrival platform, entities, and player edits.
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

    // Each byte is either native End stone (121) or air (0). This End source
    // writes no other block type before population.
    [[nodiscard]] std::array<std::uint8_t, block_count> generate_chunk(
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
