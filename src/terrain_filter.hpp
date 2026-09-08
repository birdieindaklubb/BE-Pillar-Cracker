#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pe115::terrain_filter {

// Reads a pillar-cracker result file (or a one-seed-per-line list) and an
// `x y z value` observation file.  `value` is 1 for base-terrain End stone and
// 0 for base-terrain air.  The returned seeds retain only exact PE 1.1.5 End
// base-terrain matches.
[[nodiscard]] std::vector<std::uint32_t> filter_files(
    const std::string& candidate_path,
    const std::string& observation_path,
    unsigned int worker_count = 0U);

} // namespace pe115::terrain_filter
