#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pe115::terrain_filter {

struct TerrainMismatch final {
    std::int32_t x{};
    std::int32_t y{};
    std::int32_t z{};
    bool expected_end_stone{};
    bool actual_end_stone{};
    std::uint8_t actual_block{};
};

struct TerrainVerification final {
    std::size_t observation_count{};
    std::vector<TerrainMismatch> mismatches;
};

// Reads a pillar-cracker result file (or a one-seed-per-line list) and an
// `x y z value` observation file.  `value` is 1 for End stone and 0 for
// literal air in the fresh, seed-driven End terrain.  The returned seeds
// retain only exact PE 1.1.5 matches for those observed blocks.
[[nodiscard]] std::vector<std::uint32_t> filter_files(
    const std::string& candidate_path,
    const std::string& observation_path,
    unsigned int worker_count = 0U);

// Diagnoses a terrain observation file against one world seed.  This uses the
// identical block lookup as filter_files(), while retaining every mismatch so
// a recorded coordinate can be corrected without trial-and-error filtering.
[[nodiscard]] TerrainVerification verify_file(
    std::uint32_t seed,
    const std::string& observation_path);

} // namespace pe115::terrain_filter
