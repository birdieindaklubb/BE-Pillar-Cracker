// Exact second-stage filter for the seed-driven, fresh PE 1.1.5 End terrain.
//
// This uses the cracker's intentionally narrow, self-contained PE 1.1.5 End
// terrain implementation.  It does not link a wider worldgen library.

#include "terrain_filter.hpp"

#include "pe115_end_terrain.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pe115::terrain_filter {
namespace {

constexpr std::uint8_t kPe115EndStoneBlock = 121U;
constexpr std::uint8_t kAirBlock = 0U;
constexpr std::int32_t kChunkWidth = 16;
constexpr std::int32_t kChunkHeight = 128;

struct Observation final {
    std::int32_t x{};
    std::int32_t y{};
    std::int32_t z{};
    bool end_stone{};
    std::int32_t chunk_x{};
    std::int32_t chunk_z{};
    std::int32_t local_x{};
    std::int32_t local_z{};
};

[[nodiscard]] std::string_view trim(std::string_view value) noexcept {
    while (!value.empty()
        && std::isspace(static_cast<unsigned char>(value.front())) != 0) {
        value.remove_prefix(1U);
    }
    while (!value.empty()
        && std::isspace(static_cast<unsigned char>(value.back())) != 0) {
        value.remove_suffix(1U);
    }
    return value;
}

[[nodiscard]] bool parse_seed(
    std::string_view text,
    std::uint32_t& output) noexcept {
    text = trim(text);
    if (!text.empty() && text.front() == '-') {
        std::int64_t parsed = 0;
        const auto [end, error] = std::from_chars(
            text.data(), text.data() + text.size(), parsed, 10);
        if (error != std::errc{} || end != text.data() + text.size()
            || parsed < std::numeric_limits<std::int32_t>::min()
            || parsed > std::numeric_limits<std::int32_t>::max()) {
            return false;
        }
        output = static_cast<std::uint32_t>(static_cast<std::int32_t>(parsed));
        return true;
    }

    int base = 10;
    if (text.size() >= 2U && text[0] == '0'
        && (text[1] == 'x' || text[1] == 'X')) {
        text.remove_prefix(2U);
        base = 16;
    }
    if (text.empty()) {
        return false;
    }
    std::uint64_t parsed = 0;
    const auto [end, error] = std::from_chars(
        text.data(), text.data() + text.size(), parsed, base);
    if (error != std::errc{} || end != text.data() + text.size()
        || parsed > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    output = static_cast<std::uint32_t>(parsed);
    return true;
}

[[nodiscard]] std::vector<std::uint32_t> load_candidates(
    const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot read candidate file: " + path);
    }

    std::vector<std::uint32_t> candidates;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const std::string_view view = trim(line);
        if (view.empty() || view.starts_with('#')) {
            continue;
        }

        constexpr std::string_view kOutputPrefix = "unsigned=";
        const std::size_t marker = view.find(kOutputPrefix);
        std::string_view token;
        if (marker != std::string_view::npos) {
            token = view.substr(marker + kOutputPrefix.size());
            const std::size_t end = token.find_first_of(" \t\r\n");
            token = token.substr(0U, end);
        } else {
            const std::size_t end = view.find_first_of(" \t\r\n");
            token = view.substr(0U, end);
        }

        std::uint32_t seed = 0;
        if (parse_seed(token, seed)) {
            candidates.push_back(seed);
        } else if (marker != std::string_view::npos) {
            throw std::runtime_error("Invalid unsigned seed at candidate-file "
                "line " + std::to_string(line_number) + ".");
        } else if (view.starts_with("Scanning ")
            || view.starts_with("Using the ")
            || view.starts_with("Matching full PE 1.1.5 world seeds:")
            || view.starts_with("PE 1.1.5 End terrain filter complete.")) {
            // These are the non-seed status lines printed by this executable,
            // so its stdout can be used directly as the next filter input.
            continue;
        } else {
            throw std::runtime_error("Expected one 32-bit seed at "
                "candidate-file line " + std::to_string(line_number) + ".");
        }
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
        candidates.end());
    if (candidates.empty()) {
        throw std::runtime_error("The candidate file contained no 32-bit seeds.");
    }
    return candidates;
}

void split_coordinate(
    std::int32_t coordinate,
    std::int32_t& chunk,
    std::int32_t& local) noexcept {
    chunk = coordinate / kChunkWidth;
    local = coordinate % kChunkWidth;
    if (local < 0) {
        --chunk;
        local += kChunkWidth;
    }
}

[[nodiscard]] std::vector<Observation> load_observations(
    const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot read terrain-observation file: " + path);
    }

    std::vector<Observation> observations;
    std::map<std::tuple<std::int32_t, std::int32_t, std::int32_t>, bool> seen;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (const std::size_t comment = line.find('#'); comment != std::string::npos) {
            line.resize(comment);
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        const std::string_view view = trim(line);
        if (view.empty()) {
            continue;
        }

        std::istringstream fields{std::string(view)};
        std::int64_t x = 0;
        std::int64_t y = 0;
        std::int64_t z = 0;
        std::int64_t value = 0;
        std::string extra;
        if (!(fields >> x >> y >> z >> value) || (fields >> extra)
            || x < std::numeric_limits<std::int32_t>::min()
            || x > std::numeric_limits<std::int32_t>::max()
            || z < std::numeric_limits<std::int32_t>::min()
            || z > std::numeric_limits<std::int32_t>::max()
            || y < 0 || y >= kChunkHeight || (value != 0 && value != 1)) {
            throw std::runtime_error("Expected `x y z value` at terrain-file "
                "line " + std::to_string(line_number)
                + "; value must be 0 (air) or 1 (End stone).");
        }

        const std::int32_t world_x = static_cast<std::int32_t>(x);
        const std::int32_t world_y = static_cast<std::int32_t>(y);
        const std::int32_t world_z = static_cast<std::int32_t>(z);
        Observation observation{
            .x = world_x,
            .y = world_y,
            .z = world_z,
            .end_stone = value == 1,
        };
        const auto key = std::tuple{world_x, world_y, world_z};
        const auto [found, inserted] = seen.emplace(key, observation.end_stone);
        if (!inserted) {
            if (found->second != observation.end_stone) {
                throw std::runtime_error("Conflicting observations at terrain-file "
                    "line " + std::to_string(line_number) + ".");
            }
            continue;
        }
        split_coordinate(world_x,
            observation.chunk_x, observation.local_x);
        split_coordinate(world_z,
            observation.chunk_z, observation.local_z);
        observations.push_back(observation);
    }
    if (observations.empty()) {
        throw std::runtime_error("The terrain-observation file contained no blocks.");
    }
    return observations;
}

template <typename MismatchHandler>
[[nodiscard]] bool visit_observations(
    std::uint32_t seed,
    const std::vector<Observation>& observations,
    MismatchHandler&& on_mismatch) {
    end_terrain::Generator generator(seed);
    std::map<std::pair<std::int32_t, std::int32_t>,
        std::array<std::uint8_t, end_terrain::Generator::block_count>> chunks;

    for (const Observation& observation : observations) {
        const auto key = std::pair{observation.chunk_x, observation.chunk_z};
        auto found = chunks.find(key);
        if (found == chunks.end()) {
            found = chunks.emplace(
                key,
                generator.generate_fresh_visible_chunk(
                    observation.chunk_x, observation.chunk_z)).first;
        }
        const std::size_t block_index =
            (static_cast<std::size_t>(observation.local_x) * kChunkWidth
                + static_cast<std::size_t>(observation.local_z))
            * kChunkHeight + static_cast<std::size_t>(observation.y);
        const std::uint8_t block = found->second[block_index];
        const bool actual_end_stone = block == kPe115EndStoneBlock;
        const bool matches = observation.end_stone
            ? actual_end_stone
            : block == kAirBlock;
        if (!matches) {
            if (!on_mismatch(TerrainMismatch{
                .x = observation.x,
                .y = observation.y,
                .z = observation.z,
                .expected_end_stone = observation.end_stone,
                .actual_end_stone = actual_end_stone,
                .actual_block = block,
            })) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] bool matches_observations(
    std::uint32_t seed,
    const std::vector<Observation>& observations) {
    return visit_observations(seed, observations,
        [](const TerrainMismatch&) { return false; });
}

[[nodiscard]] TerrainVerification verify_observations(
    std::uint32_t seed,
    const std::vector<Observation>& observations) {
    TerrainVerification result;
    result.observation_count = observations.size();
    static_cast<void>(visit_observations(seed, observations,
        [&result](TerrainMismatch mismatch) {
            result.mismatches.push_back(mismatch);
            return true;
        }));
    return result;
}

} // namespace

std::vector<std::uint32_t> filter_files(
    const std::string& candidate_path,
    const std::string& observation_path,
    unsigned int worker_count) {
    const auto candidates = load_candidates(candidate_path);
    const auto observations = load_observations(observation_path);

    // Candidate worlds are independent. Keep the acceptance flags indexed by
    // their already-sorted input position, then rebuild the result serially;
    // this gives a deterministic output order without synchronizing every
    // accepted seed.
    std::vector<unsigned char> accepted(candidates.size(), 0U);
#ifdef _OPENMP
    if (worker_count != 0U) {
        omp_set_num_threads(static_cast<int>(worker_count));
    }
#pragma omp parallel for schedule(static)
    for (std::int64_t index = 0;
         index < static_cast<std::int64_t>(candidates.size());
         ++index) {
        accepted[static_cast<std::size_t>(index)] = matches_observations(
            candidates[static_cast<std::size_t>(index)], observations)
            ? 1U : 0U;
    }
#else
    static_cast<void>(worker_count);
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        accepted[index] = matches_observations(candidates[index], observations)
            ? 1U : 0U;
    }
#endif

    std::vector<std::uint32_t> result;
    result.reserve(candidates.size());
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        if (accepted[index] != 0U) {
            result.push_back(candidates[index]);
        }
    }
    return result;
}

TerrainVerification verify_file(
    std::uint32_t seed,
    const std::string& observation_path) {
    return verify_observations(seed, load_observations(observation_path));
}

} // namespace pe115::terrain_filter
