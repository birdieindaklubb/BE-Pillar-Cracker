// Standalone PE 1.1.5.0 End-pillar seed candidate scanner.
//
// This source is an independently authored behavioral reimplementation.  It
// intentionally contains only the MT19937 prefix and shuffle needed to model
// the observed End-pillar height layout, not a general world generator.

#include "pillar_constraints.hpp"
#include "pillar_layout.hpp"
#include "avx2.hpp"
#include "pe115_end_terrain.hpp"
#ifdef PE115_HAVE_CUDA
#include "cuda_backend.hpp"
#endif
#include "terrain_filter.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr std::size_t kPillarCount = pe115::kPillarCount;
constexpr std::size_t kShuffleDrawCount = kPillarCount - 1;
constexpr const auto& kPillarSites = pe115::pillars::sites;

using Observations = pe115::PillarShapeMasks;

enum class ScanKind {
    none,
    all,
    contiguous_range,
    high_word,
    low_word,
};

struct Options final {
    Observations observations{pe115::unconstrained_pillars()};
    ScanKind scan_kind = ScanKind::none;
    std::uint64_t range_start = 0;
    std::uint64_t range_count = 0;
    std::uint16_t word = 0;
    std::optional<std::uint32_t> verify_seed;
    std::optional<std::pair<std::string, std::string>> terrain_filter_files;
    std::optional<std::pair<std::uint32_t, std::string>> terrain_verify;
    unsigned int threads = 0;
    bool scalar = false;
    bool avx2_requested = false;
    bool cuda_requested = false;
    bool self_test = false;
};

enum class SelectedBackend {
    scalar,
    avx2,
    cuda,
};

[[nodiscard]] constexpr std::uint16_t shape_bit(
    std::uint8_t shape) noexcept {
    return static_cast<std::uint16_t>(1U << shape);
}

[[nodiscard]] constexpr bool is_single_shape(
    std::uint16_t mask) noexcept {
    return mask != 0U && (mask & static_cast<std::uint16_t>(mask - 1U)) == 0U;
}

[[nodiscard]] std::uint8_t shape_from_mask(std::uint16_t mask) noexcept {
    for (std::uint8_t shape = 0U; shape < kPillarCount; ++shape) {
        if (mask == shape_bit(shape)) {
            return shape;
        }
    }
    return 0U;
}

[[nodiscard]] std::uint16_t radius_mask(std::int32_t radius) {
    if (radius < 2 || radius > 5) {
        throw std::runtime_error("Pillar radius must be 2, 3, 4, or 5.");
    }
    std::uint16_t result = 0U;
    for (std::uint8_t shape = 0U; shape < kPillarCount; ++shape) {
        if (pe115::pillars::radius_for_shape(shape) == radius) {
            result = static_cast<std::uint16_t>(result | shape_bit(shape));
        }
    }
    return result;
}

[[nodiscard]] constexpr std::uint16_t cage_mask(bool caged) noexcept {
    constexpr std::uint16_t caged_shapes = shape_bit(1U) | shape_bit(2U);
    return caged ? caged_shapes
                 : static_cast<std::uint16_t>(
                       pe115::kAllPillarShapes & ~caged_shapes);
}

[[nodiscard]] bool matches(
    std::uint32_t seed,
    const Observations& observations) noexcept {
    const auto shapes = pe115::pillars::shapes(seed);
    for (std::size_t index = 0; index < shapes.size(); ++index) {
        if ((observations[index] & shape_bit(shapes[index])) == 0U) {
            return false;
        }
    }
    return true;
}

using DrawConstraints = std::array<std::uint8_t, kShuffleDrawCount>;

// Invert the forward Fisher-Yates swaps represented by a completely observed
// final pillar order.  This turns the height permutation into nine independent
// MT output modulo constraints, which is equivalent to running the shuffle
// forward but cheaper to test and suitable for the vectorized scanner.
[[nodiscard]] DrawConstraints derive_draws(
    const Observations& observations) {
    std::array<std::uint8_t, kPillarCount> order{};
    for (std::size_t index = 0; index < order.size(); ++index) {
        if (!is_single_shape(observations[index])) {
            throw std::runtime_error("Cannot derive shuffle constraints from "
                "a partially constrained pillar layout.");
        }
        order[index] = shape_from_mask(observations[index]);
    }

    DrawConstraints result{};
    for (std::size_t index = kPillarCount - 1; index > 0; --index) {
        std::size_t selected = 0;
        while (selected <= index && order[selected] != index) {
            ++selected;
        }
        if (selected > index) {
            throw std::runtime_error("The supplied heights are not a valid "
                "pillar permutation.");
        }
        result[index - 1] = static_cast<std::uint8_t>(selected);
        std::swap(order[selected], order[index]);
    }
    return result;
}

[[nodiscard]] bool matches_draws(
    std::uint32_t seed,
    const DrawConstraints& draws) noexcept {
    std::array<std::uint32_t, 10> initial_low{};
    initial_low[0] = seed;
    std::uint32_t high_state = seed;
    for (std::uint32_t state_index = 1; state_index <= 397U; ++state_index) {
        high_state = pe115::pillars::seed_step(high_state, state_index);
        if (state_index < initial_low.size()) {
            initial_low[state_index] = high_state;
        }
    }

    for (std::size_t index = 0; index < draws.size(); ++index) {
        const std::uint32_t joined =
            (initial_low[index] & pe115::pillars::mt_upper_mask)
            | (initial_low[index + 1U] & pe115::pillars::mt_lower_mask);
        std::uint32_t twisted = high_state ^ (joined >> 1U);
        if ((joined & 1U) != 0U) {
            twisted ^= pe115::pillars::mt_matrix_a;
        }
        if (pe115::pillars::temper(twisted)
            % static_cast<std::uint32_t>(index + 2U)
            != draws[index]) {
            return false;
        }
        if (index + 1U != draws.size()) {
            high_state = pe115::pillars::seed_step(
                high_state, static_cast<std::uint32_t>(398U + index));
        }
    }
    return true;
}

[[nodiscard]] std::uint64_t parse_unsigned(
    std::string_view text,
    std::uint64_t maximum,
    std::string_view label) {
    if (text.empty()) {
        throw std::runtime_error(std::string(label) + " cannot be empty.");
    }

    std::size_t consumed = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(std::string(text), &consumed, 0);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid " + std::string(label) + ": "
            + std::string(text));
    }
    if (consumed != text.size() || value > maximum) {
        throw std::runtime_error("Out-of-range " + std::string(label) + ": "
            + std::string(text));
    }
    return static_cast<std::uint64_t>(value);
}

[[nodiscard]] std::int32_t parse_i32(
    std::string_view text,
    std::string_view label) {
    std::int32_t value = 0;
    const auto [end, error] = std::from_chars(
        text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::runtime_error("Invalid " + std::string(label) + ": "
            + std::string(text));
    }
    return value;
}

[[nodiscard]] std::vector<std::string_view> split_csv(std::string_view text) {
    std::vector<std::string_view> fields;
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t end = text.find(',', start);
        const std::size_t length = end == std::string_view::npos
            ? text.size() - start
            : end - start;
        fields.emplace_back(text.substr(start, length));
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }
    return fields;
}

[[nodiscard]] std::uint8_t feature_height_to_shape(
    std::int32_t height,
    std::string_view label) {
    constexpr std::int32_t kMinimumHeight = 76;
    constexpr std::int32_t kHeightStep = 3;
    const std::int32_t shifted = height - kMinimumHeight;
    if (shifted < 0 || shifted % kHeightStep != 0
        || shifted / kHeightStep >= static_cast<std::int32_t>(kPillarCount)) {
        throw std::runtime_error(std::string(label) + " must be one of 76, 79, "
            "82, 85, 88, 91, 94, 97, 100, or 103.");
    }
    return static_cast<std::uint8_t>(shifted / kHeightStep);
}

void add_constraint(
    Observations& observations,
    std::size_t index,
    std::uint16_t allowed_shapes,
    std::string_view label) {
    const std::uint16_t combined = static_cast<std::uint16_t>(
        observations[index] & allowed_shapes);
    if (combined == 0U) {
        throw std::runtime_error("Conflicting " + std::string(label)
            + " supplied for pillar index " + std::to_string(index) + ".");
    }
    observations[index] = combined;
}

void set_heights(
    Observations& observations,
    std::string_view csv,
    bool obsidian_tops) {
    const auto fields = split_csv(csv);
    if (fields.size() != kPillarCount) {
        throw std::runtime_error("--heights and --obsidian-tops need exactly "
            "ten comma-separated values.");
    }
    for (std::size_t index = 0; index < fields.size(); ++index) {
        const std::int32_t measured = parse_i32(fields[index], "height");
        const std::int32_t feature_height = obsidian_tops ? measured + 1 : measured;
        add_constraint(
            observations,
            index,
            shape_bit(feature_height_to_shape(feature_height, "Pillar height")),
            "height");
    }
}

[[nodiscard]] std::size_t pillar_index(
    std::int32_t x,
    std::int32_t z) {
    for (std::size_t index = 0; index < kPillarSites.size(); ++index) {
        if (kPillarSites[index].x == x && kPillarSites[index].z == z) {
            return index;
        }
    }
    throw std::runtime_error("The supplied pillar coordinate is not a PE 1.1.5 "
        "End ring center.");
}

[[nodiscard]] bool parse_caged(
    std::string_view text) {
    std::string normalized;
    normalized.reserve(text.size());
    for (const unsigned char character : text) {
        normalized.push_back(static_cast<char>(std::tolower(character)));
    }
    if (normalized == "caged" || normalized == "cage"
        || normalized == "yes" || normalized == "true" || normalized == "1") {
        return true;
    }
    if (normalized == "uncaged" || normalized == "no-cage"
        || normalized == "no" || normalized == "false" || normalized == "0") {
        return false;
    }
    throw std::runtime_error("Pillar cage must be CAGED or UNCAGED.");
}

void add_radius_constraint(
    Observations& observations,
    std::size_t index,
    std::int32_t radius) {
    add_constraint(observations, index, radius_mask(radius), "radius");
}

void add_cage_constraint(
    Observations& observations,
    std::size_t index,
    bool caged) {
    add_constraint(observations, index, cage_mask(caged), "cage state");
}

void set_pillar(
    Observations& observations,
    std::string_view csv,
    bool obsidian_tops) {
    const auto fields = split_csv(csv);
    if (fields.size() < 3U || fields.size() > 5U) {
        throw std::runtime_error("--pillar must have the form "
            "X,Z,HEIGHT[,RADIUS[,CAGED|UNCAGED]].");
    }
    const std::int32_t x = parse_i32(fields[0], "pillar X");
    const std::int32_t z = parse_i32(fields[1], "pillar Z");
    const std::int32_t measured_height = parse_i32(fields[2], "pillar height");
    if (obsidian_tops && measured_height == std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error("Pillar top is out of range.");
    }
    const std::int32_t feature_height = obsidian_tops
        ? measured_height + 1
        : measured_height;
    const std::size_t index = pillar_index(x, z);
    add_constraint(observations, index,
        shape_bit(feature_height_to_shape(feature_height, "Pillar height")), "height");
    if (fields.size() >= 4U) {
        add_radius_constraint(
            observations, index, parse_i32(fields[3], "pillar radius"));
    }
    if (fields.size() == 5U) {
        add_cage_constraint(observations, index, parse_caged(fields[4]));
    }
}

void set_pillar_radius(
    Observations& observations,
    std::string_view csv) {
    const auto fields = split_csv(csv);
    if (fields.size() != 3U) {
        throw std::runtime_error("--pillar-radius must have the form X,Z,RADIUS.");
    }
    const std::size_t index = pillar_index(
        parse_i32(fields[0], "pillar X"), parse_i32(fields[1], "pillar Z"));
    add_radius_constraint(
        observations, index, parse_i32(fields[2], "pillar radius"));
}

void set_pillar_cage(
    Observations& observations,
    std::string_view csv) {
    const auto fields = split_csv(csv);
    if (fields.size() != 3U) {
        throw std::runtime_error("--pillar-cage must have the form "
            "X,Z,CAGED|UNCAGED.");
    }
    const std::size_t index = pillar_index(
        parse_i32(fields[0], "pillar X"), parse_i32(fields[1], "pillar Z"));
    add_cage_constraint(observations, index, parse_caged(fields[2]));
}

[[nodiscard]] std::size_t observation_count(
    const Observations& observations) noexcept {
    return static_cast<std::size_t>(std::count_if(
        observations.begin(), observations.end(),
        [](std::uint16_t value) { return value != pe115::kAllPillarShapes; }));
}

void validate_unique_shapes(const Observations& observations) {
    std::array<bool, kPillarCount> used{};
    for (const std::uint16_t mask : observations) {
        if (!is_single_shape(mask)) {
            continue;
        }
        const std::uint8_t shape = shape_from_mask(mask);
        if (used[shape]) {
            throw std::runtime_error("Each exact pillar-shape constraint must "
                "be unique. The ten PE pillar shapes form one permutation.");
        }
        used[shape] = true;
    }
}

[[nodiscard]] bool is_complete_permutation(
    const Observations& observations) noexcept {
    return std::all_of(observations.begin(), observations.end(),
        [](std::uint16_t mask) { return is_single_shape(mask); });
}

[[nodiscard]] std::string next_argument(
    int& index,
    int argc,
    char* argv[],
    std::string_view flag) {
    if (++index >= argc) {
        throw std::runtime_error(std::string(flag) + " needs a value.");
    }
    return argv[index];
}

[[nodiscard]] Options parse_options(int argc, char* argv[]) {
    Options options;
    bool have_range_start = false;
    bool have_range_count = false;

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--heights") {
            set_heights(options.observations,
                next_argument(index, argc, argv, argument), false);
        } else if (argument == "--obsidian-tops") {
            set_heights(options.observations,
                next_argument(index, argc, argv, argument), true);
        } else if (argument == "--pillar") {
            set_pillar(options.observations,
                next_argument(index, argc, argv, argument), false);
        } else if (argument == "--pillar-top") {
            set_pillar(options.observations,
                next_argument(index, argc, argv, argument), true);
        } else if (argument == "--pillar-radius") {
            set_pillar_radius(options.observations,
                next_argument(index, argc, argv, argument));
        } else if (argument == "--pillar-cage") {
            set_pillar_cage(options.observations,
                next_argument(index, argc, argv, argument));
        } else if (argument == "--high16") {
            if (options.scan_kind != ScanKind::none) {
                throw std::runtime_error("Choose only one scan selector.");
            }
            options.word = static_cast<std::uint16_t>(parse_unsigned(
                next_argument(index, argc, argv, argument), 0xffffU, "high16"));
            options.scan_kind = ScanKind::high_word;
        } else if (argument == "--low16") {
            if (options.scan_kind != ScanKind::none) {
                throw std::runtime_error("Choose only one scan selector.");
            }
            options.word = static_cast<std::uint16_t>(parse_unsigned(
                next_argument(index, argc, argv, argument), 0xffffU, "low16"));
            options.scan_kind = ScanKind::low_word;
        } else if (argument == "--range-start") {
            if (have_range_start) {
                throw std::runtime_error("--range-start was provided twice.");
            }
            options.range_start = parse_unsigned(next_argument(
                index, argc, argv, argument), 0xffff'ffffULL, "range start");
            have_range_start = true;
        } else if (argument == "--count") {
            if (have_range_count) {
                throw std::runtime_error("--count was provided twice.");
            }
            options.range_count = parse_unsigned(next_argument(
                index, argc, argv, argument), 0x1'0000'0000ULL, "count");
            have_range_count = true;
        } else if (argument == "--all") {
            if (options.scan_kind != ScanKind::none || have_range_start
                || have_range_count) {
                throw std::runtime_error("Choose only one scan selector.");
            }
            options.scan_kind = ScanKind::all;
        } else if (argument == "--verify") {
            if (options.verify_seed.has_value()) {
                throw std::runtime_error("--verify was provided twice.");
            }
            options.verify_seed = static_cast<std::uint32_t>(parse_unsigned(
                next_argument(index, argc, argv, argument), 0xffff'ffffULL, "seed"));
        } else if (argument == "--terrain-filter") {
            if (options.terrain_filter_files.has_value()) {
                throw std::runtime_error("--terrain-filter was provided twice.");
            }
            const std::string candidates = next_argument(
                index, argc, argv, argument);
            const std::string terrain = next_argument(index, argc, argv, argument);
            options.terrain_filter_files = {candidates, terrain};
        } else if (argument == "--terrain-verify") {
            if (options.terrain_verify.has_value()) {
                throw std::runtime_error("--terrain-verify was provided twice.");
            }
            const std::uint32_t seed = static_cast<std::uint32_t>(parse_unsigned(
                next_argument(index, argc, argv, argument), 0xffff'ffffULL, "seed"));
            options.terrain_verify = {seed, next_argument(index, argc, argv, argument)};
        } else if (argument == "--threads") {
            const auto requested = parse_unsigned(next_argument(
                index, argc, argv, argument),
                static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
                "thread count");
            if (requested == 0U) {
                throw std::runtime_error("--threads must be at least one.");
            }
            options.threads = static_cast<unsigned int>(requested);
        } else if (argument == "--scalar") {
            options.scalar = true;
        } else if (argument == "--avx2") {
            options.avx2_requested = true;
        } else if (argument == "--cuda") {
            options.cuda_requested = true;
        } else if (argument == "--self-test") {
            options.self_test = true;
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage:\n"
                << "  pe115_pillarcracker --heights H0,...,H9 --high16 WORD\n"
                << "  pe115_pillarcracker --pillar X,Z,H ... --low16 WORD\n"
                << "  pe115_pillarcracker --heights H0,...,H9 "
                   "--range-start START --count COUNT\n"
                << "  pe115_pillarcracker --heights H0,...,H9 --all\n"
                << "  pe115_pillarcracker --verify SEED [--heights H0,...,H9]\n\n"
                << "  --pillar X,Z,H[,R[,CAGED|UNCAGED]]\n"
                << "  --pillar-top X,Z,TOP[,R[,CAGED|UNCAGED]]\n"
                << "  --pillar-radius X,Z,R  --pillar-cage X,Z,CAGED|UNCAGED\n\n"
                << "  pe115_pillarcracker --terrain-filter CANDIDATES.txt TERRAIN.txt\n"
                << "  pe115_pillarcracker --terrain-verify SEED TERRAIN.txt\n\n"
                << "Heights are feature/crystal-layer Y values 76..103 in steps "
                   "of 3.  --cuda/--avx2 require those backends; --scalar "
                   "disables them.  "
                   "See README.md for the fixed PE ring order.\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error("Unknown option: " + std::string(argument));
        }
    }

    if (have_range_start != have_range_count) {
        throw std::runtime_error("--range-start and --count must be used together.");
    }
    if (have_range_start) {
        if (options.scan_kind != ScanKind::none) {
            throw std::runtime_error("Choose only one scan selector.");
        }
        if (options.range_count == 0U || options.range_start + options.range_count
            > 0x1'0000'0000ULL) {
            throw std::runtime_error("The requested range must stay within the "
                "32-bit seed space and have a non-zero count.");
        }
        options.scan_kind = ScanKind::contiguous_range;
    }
    return options;
}

[[nodiscard]] std::uint64_t scan_count(const Options& options) noexcept {
    switch (options.scan_kind) {
    case ScanKind::all:
        return 0x1'0000'0000ULL;
    case ScanKind::contiguous_range:
        return options.range_count;
    case ScanKind::high_word:
    case ScanKind::low_word:
        return 0x1'0000ULL;
    case ScanKind::none:
        return 0U;
    }
    return 0U;
}

[[nodiscard]] std::uint32_t seed_at(
    const Options& options,
    std::uint64_t index) noexcept {
    switch (options.scan_kind) {
    case ScanKind::all:
        return static_cast<std::uint32_t>(index);
    case ScanKind::contiguous_range:
        return static_cast<std::uint32_t>(options.range_start + index);
    case ScanKind::high_word:
        return (static_cast<std::uint32_t>(options.word) << 16U)
            | static_cast<std::uint32_t>(index);
    case ScanKind::low_word:
        return (static_cast<std::uint32_t>(index) << 16U) | options.word;
    case ScanKind::none:
        return 0U;
    }
    return 0U;
}

[[nodiscard]] std::string hex_value(std::uint32_t value, int width) {
    std::ostringstream stream;
    stream << "0x" << std::uppercase << std::hex << std::setfill('0')
           << std::setw(width) << value;
    return stream.str();
}

void print_layout(std::uint32_t seed) {
    const auto shapes = pe115::pillars::shapes(seed);
    std::cout << "Seed " << seed << " (signed "
              << static_cast<std::int32_t>(seed) << ", "
              << hex_value(seed, 8) << ")\n";
    std::cout << "index  center       feature-height  top-obsidian  radius  cage\n";
    for (std::size_t index = 0; index < kPillarCount; ++index) {
        const std::int32_t height = 76 + 3 * static_cast<std::int32_t>(shapes[index]);
        std::cout << std::setw(5) << index << "  ("
                  << std::setw(3) << kPillarSites[index].x << ","
                  << std::setw(3) << kPillarSites[index].z << ")"
                  << std::setw(12) << height
                  << std::setw(14) << height - 1
                  << std::setw(8) << static_cast<int>(
                         pe115::pillars::radius_for_shape(shapes[index]))
                  << std::setw(7)
                  << (pe115::pillars::is_caged_shape(shapes[index])
                          ? "caged" : "none")
                  << "\n";
    }
}

[[nodiscard]] bool self_test() {
    constexpr std::array<std::uint32_t, kShuffleDrawCount> expected{{
        3'499'211'612U, 581'869'302U, 3'890'346'734U,
        3'586'334'585U, 545'404'204U, 4'161'255'391U,
        3'922'919'429U, 949'333'985U, 2'715'962'298U,
    }};
    if (pe115::pillars::shuffle_words(5489U) != expected) {
        std::cerr << "Self-test failed: MT19937 prefix mismatch.\n";
        return false;
    }

    constexpr std::uint32_t test_seed = 0x1234'5678U;
    Observations observation{pe115::unconstrained_pillars()};
    const auto shapes = pe115::pillars::shapes(test_seed);
    for (std::size_t index = 0; index < shapes.size(); ++index) {
        observation[index] = shape_bit(shapes[index]);
    }
    if (!matches(test_seed, observation)) {
        std::cerr << "Self-test failed: matching layout rejected.\n";
        return false;
    }
    const DrawConstraints draws = derive_draws(observation);
    if (!matches_draws(test_seed, draws)) {
        std::cerr << "Self-test failed: inverted shuffle constraints rejected.\n";
        return false;
    }
    observation[0] = shape_bit(static_cast<std::uint8_t>(
        (shapes[0] + 1U) % kPillarCount));
    if (matches(test_seed, observation)) {
        std::cerr << "Self-test failed: altered layout accepted.\n";
        return false;
    }
    Observations feature_observation{pe115::unconstrained_pillars()};
    for (std::size_t index = 0; index < shapes.size(); ++index) {
        add_radius_constraint(feature_observation, index,
            pe115::pillars::radius_for_shape(shapes[index]));
        add_cage_constraint(feature_observation, index,
            pe115::pillars::is_caged_shape(shapes[index]));
    }
    if (!matches(test_seed, feature_observation)) {
        std::cerr << "Self-test failed: native radius/cage layout rejected.\n";
        return false;
    }
    std::cout << "Self-test passed: standard MT19937 prefix and PE pillar "
                 "shuffle checks succeeded.\n";
    return true;
}

[[nodiscard]] pe115::avx2::ScanSpec avx2_specification(
    const Options& options) noexcept {
    pe115::avx2::ScanSpec result{
        .kind = pe115::avx2::ScanKind::all,
        .count = scan_count(options),
        .range_start = options.range_start,
        .word = options.word,
    };
    switch (options.scan_kind) {
    case ScanKind::all:
        result.kind = pe115::avx2::ScanKind::all;
        break;
    case ScanKind::contiguous_range:
        result.kind = pe115::avx2::ScanKind::contiguous_range;
        break;
    case ScanKind::high_word:
        result.kind = pe115::avx2::ScanKind::high_word;
        break;
    case ScanKind::low_word:
        result.kind = pe115::avx2::ScanKind::low_word;
        break;
    case ScanKind::none:
        break;
    }
    return result;
}

[[nodiscard]] SelectedBackend select_backend(const Options& options) {
    const unsigned int requested = static_cast<unsigned int>(options.scalar)
        + static_cast<unsigned int>(options.avx2_requested)
        + static_cast<unsigned int>(options.cuda_requested);
    if (requested > 1U) {
        throw std::runtime_error("Choose only one of --scalar, --avx2, or "
            "--cuda.");
    }
    if (options.scalar) {
        return SelectedBackend::scalar;
    }

#ifdef PE115_HAVE_AVX2
    if (options.avx2_requested) {
        if (pe115::avx2::is_available()) {
            return SelectedBackend::avx2;
        }
        throw std::runtime_error("AVX2 was requested but this machine cannot "
            "run the AVX2 backend.");
    }
#else
    if (options.avx2_requested) {
        throw std::runtime_error("AVX2 was requested but this build has no "
            "AVX2 backend.");
    }
#endif

#ifdef PE115_HAVE_CUDA
    if (pe115::cuda_backend::is_available()) {
        return SelectedBackend::cuda;
    }
#endif
    if (options.cuda_requested) {
        throw std::runtime_error("CUDA was requested but this build has no "
            "usable CUDA device.");
    }

#ifdef PE115_HAVE_AVX2
    if (pe115::avx2::is_available()) {
        return SelectedBackend::avx2;
    }
#endif
    return SelectedBackend::scalar;
}

[[nodiscard]] const char* backend_label(SelectedBackend backend) noexcept {
    switch (backend) {
    case SelectedBackend::scalar:
        return "portable scalar";
    case SelectedBackend::avx2:
        return "exact eight-lane AVX2";
    case SelectedBackend::cuda:
        return "exact CUDA";
    }
    return "unknown";
}

[[nodiscard]] std::vector<std::uint32_t> scan(
    const Options& options,
    const DrawConstraints& draws,
    SelectedBackend backend) {
    const std::uint64_t total = scan_count(options);
    std::vector<std::uint32_t> candidates;

#ifdef _OPENMP
    omp_set_num_threads(static_cast<int>(options.threads));
#endif

#ifdef PE115_HAVE_CUDA
    if (backend == SelectedBackend::cuda) {
        candidates = pe115::cuda_backend::scan(avx2_specification(options), draws);
        std::sort(candidates.begin(), candidates.end());
        return candidates;
    }
#endif

#ifdef PE115_HAVE_AVX2
    if (backend == SelectedBackend::avx2) {
        candidates = pe115::avx2::scan(avx2_specification(options), draws);
        std::sort(candidates.begin(), candidates.end());
        return candidates;
    }
#endif

#ifdef _OPENMP
#pragma omp parallel
    {
        std::vector<std::uint32_t> local_candidates;
#pragma omp for schedule(static)
        for (long long index = 0; index < static_cast<long long>(total); ++index) {
            const std::uint32_t seed = seed_at(
                options, static_cast<std::uint64_t>(index));
            if (matches_draws(seed, draws)) {
                local_candidates.push_back(seed);
            }
        }
#pragma omp critical
        candidates.insert(
            candidates.end(), local_candidates.begin(), local_candidates.end());
    }
#else
    for (std::uint64_t index = 0; index < total; ++index) {
        const std::uint32_t seed = seed_at(options, index);
        if (matches_draws(seed, draws)) {
            candidates.push_back(seed);
        }
    }
#endif

    std::sort(candidates.begin(), candidates.end());
    return candidates;
}

[[nodiscard]] std::vector<std::uint32_t> scan(
    const Options& options,
    const Observations& observations,
    SelectedBackend backend) {
    const std::uint64_t total = scan_count(options);
    std::vector<std::uint32_t> candidates;

#ifdef _OPENMP
    omp_set_num_threads(static_cast<int>(options.threads));
#endif

#ifdef PE115_HAVE_CUDA
    if (backend == SelectedBackend::cuda) {
        candidates = pe115::cuda_backend::scan(
            avx2_specification(options), observations);
        std::sort(candidates.begin(), candidates.end());
        return candidates;
    }
#endif

#ifdef PE115_HAVE_AVX2
    if (backend == SelectedBackend::avx2) {
        candidates = pe115::avx2::scan(
            avx2_specification(options), observations);
        std::sort(candidates.begin(), candidates.end());
        return candidates;
    }
#endif

#ifdef _OPENMP
#pragma omp parallel
    {
        std::vector<std::uint32_t> local_candidates;
#pragma omp for schedule(static)
        for (long long index = 0; index < static_cast<long long>(total); ++index) {
            const std::uint32_t seed = seed_at(
                options, static_cast<std::uint64_t>(index));
            if (matches(seed, observations)) {
                local_candidates.push_back(seed);
            }
        }
#pragma omp critical
        candidates.insert(
            candidates.end(), local_candidates.begin(), local_candidates.end());
    }
#else
    for (std::uint64_t index = 0; index < total; ++index) {
        const std::uint32_t seed = seed_at(options, index);
        if (matches(seed, observations)) {
            candidates.push_back(seed);
        }
    }
#endif

    std::sort(candidates.begin(), candidates.end());
    return candidates;
}

void write_candidates(
    std::ostream& output,
    const std::vector<std::uint32_t>& candidates) {
    output << "Matching full PE 1.1.5 world seeds: " << candidates.size()
           << "\n";
    for (const std::uint32_t seed : candidates) {
        const std::uint16_t high = static_cast<std::uint16_t>(seed >> 16U);
        const std::uint16_t low = static_cast<std::uint16_t>(seed);
        output << "  unsigned=" << seed
               << "  signed=" << static_cast<std::int32_t>(seed)
               << "  hex=" << hex_value(seed, 8)
               << "  high16=" << hex_value(high, 4)
               << "  low16=" << hex_value(low, 4) << "\n";
    }
}

void print_candidates(const std::vector<std::uint32_t>& candidates) {
    write_candidates(std::cout, candidates);
}

[[nodiscard]] std::filesystem::path save_candidates(
    const std::vector<std::uint32_t>& candidates,
    std::string_view source) {
    namespace fs = std::filesystem;
    const auto now = std::chrono::system_clock::now();
    const std::time_t calendar = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
#ifdef _WIN32
    if (localtime_s(&local_time, &calendar) != 0) {
        throw std::runtime_error("Cannot read the local machine time.");
    }
#else
    if (localtime_r(&calendar, &local_time) == nullptr) {
        throw std::runtime_error("Cannot read the local machine time.");
    }
#endif
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    const auto milliseconds = static_cast<unsigned int>(
        (elapsed % 1000LL + 1000LL) % 1000LL);

    std::ostringstream id;
    id << std::put_time(&local_time, "%y%m%d-%H%M%S")
       << '-' << std::setfill('0') << std::setw(3) << milliseconds;
    const fs::path directory{"results"};
    std::error_code error;
    if (!fs::create_directories(directory, error) && error) {
        throw std::runtime_error("Cannot create results directory: "
            + error.message());
    }

    for (unsigned int collision = 0U; collision < 10'000U; ++collision) {
        std::ostringstream filename;
        filename << "pe115-" << id.str();
        if (collision != 0U) {
            filename << '-' << std::setw(2) << std::setfill('0') << collision;
        }
        filename << ".txt";
        const fs::path path = directory / filename.str();
        if (fs::exists(path, error)) {
            if (error) {
                throw std::runtime_error("Cannot inspect results directory: "
                    + error.message());
            }
            continue;
        }
        if (error) {
            throw std::runtime_error("Cannot inspect results directory: "
                + error.message());
        }

        std::ofstream output(path, std::ios::out | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("Cannot create result file: "
                + path.string());
        }
        output << "# PE 1.1.5 End pillar cracker " << source << " result\n"
               << "# Local time id: " << id.str() << "\n";
        write_candidates(output, candidates);
        if (!output) {
            throw std::runtime_error("Cannot write result file: "
                + path.string());
        }
        return path;
    }
    throw std::runtime_error("Could not allocate a unique result-file name.");
}

void print_and_save_candidates(
    const std::vector<std::uint32_t>& candidates,
    std::string_view source) {
    print_candidates(candidates);
    const std::filesystem::path path = save_candidates(candidates, source);
    std::cout << "Saved candidate list: " << path.string() << "\n";
}

[[nodiscard]] std::string describe_end_block(std::uint8_t block) {
    if (block == 0U) {
        return "air";
    }
    if (block == 121U) {
        return "End stone";
    }
    return "block " + std::to_string(block);
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        Options options = parse_options(argc, argv);
        if (options.terrain_verify.has_value()) {
            if (options.terrain_filter_files.has_value()
                || options.scan_kind != ScanKind::none
                || options.verify_seed.has_value()
                || options.self_test
                || options.scalar
                || options.avx2_requested
                || options.cuda_requested
                || observation_count(options.observations) != 0U) {
                throw std::runtime_error("--terrain-verify cannot be combined "
                    "with pillar scanning, --verify, --self-test, "
                    "--terrain-filter, or a scanner-backend selector.");
            }
            const auto& [seed, terrain_path] = *options.terrain_verify;
            const auto verification = pe115::terrain_filter::verify_file(
                seed, terrain_path);
            const std::size_t matches = verification.observation_count
                - verification.mismatches.size();
            std::cout << "PE 1.1.5 End terrain verification for unsigned="
                      << seed << ": " << matches << "/"
                      << verification.observation_count << " observations match.\n";
            for (const auto& mismatch : verification.mismatches) {
                std::cout << "  x=" << mismatch.x << " y=" << mismatch.y
                          << " z=" << mismatch.z << ": expected "
                          << (mismatch.expected_end_stone ? "End stone" : "air")
                          << ", got "
                          << describe_end_block(mismatch.actual_block)
                          << ".\n";
            }
            return verification.mismatches.empty() ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        if (options.terrain_filter_files.has_value()) {
            if (options.scan_kind != ScanKind::none
                || options.verify_seed.has_value()
                || options.self_test
                || options.terrain_verify.has_value()
                || options.scalar
                || options.avx2_requested
                || options.cuda_requested
                || observation_count(options.observations) != 0U) {
                throw std::runtime_error("--terrain-filter cannot be combined "
                    "with pillar scanning, --verify, --self-test, or a "
                    "scanner-backend selector.");
            }
            const auto& [candidate_path, terrain_path] =
                *options.terrain_filter_files;
            const auto candidates = pe115::terrain_filter::filter_files(
                candidate_path, terrain_path, options.threads);
            std::cout << "PE 1.1.5 End terrain filter complete.\n";
            print_and_save_candidates(candidates, "terrain-filter");
            return EXIT_SUCCESS;
        }
        if (options.self_test) {
            if (!self_test()) {
                return EXIT_FAILURE;
            }
            if (!pe115::end_terrain::self_test()) {
                std::cerr << "Self-test failed: PE 1.1.5 End terrain regression.\n";
                return EXIT_FAILURE;
            }
            std::cout << "Self-test passed: PE 1.1.5 End terrain regressions succeeded.\n";
        }

        validate_unique_shapes(options.observations);

        if (options.verify_seed.has_value()) {
            print_layout(*options.verify_seed);
            if (observation_count(options.observations) != 0U) {
                std::cout << "Supplied observations: "
                          << (matches(*options.verify_seed, options.observations)
                                  ? "match.\n"
                                  : "do not match.\n");
            }
        }

        if (options.scan_kind == ScanKind::none) {
            if (!options.self_test && !options.verify_seed.has_value()) {
                throw std::runtime_error("Supply a scan selector, --verify, or "
                    "--self-test.  Use --help for usage.");
            }
            return EXIT_SUCCESS;
        }

        if (observation_count(options.observations) == 0U) {
            throw std::runtime_error("Scanning needs at least one pillar height, "
                "radius, or cage observation.");
        }
        const bool complete_permutation = is_complete_permutation(
            options.observations);

        if (options.threads == 0U) {
            options.threads = std::max(1U, std::thread::hardware_concurrency());
        }
        const SelectedBackend backend = select_backend(options);

        if (backend == SelectedBackend::cuda) {
            std::cout << "Scanning " << scan_count(options)
                      << " seeds on CUDA device 0.\n";
        } else {
#ifdef _OPENMP
            std::cout << "Scanning " << scan_count(options) << " seeds with up to "
                      << options.threads << " worker(s).\n";
#else
            if (options.threads != 1U) {
                std::cout << "OpenMP was not available at build time; scanning with "
                             "one worker.\n";
            }
            std::cout << "Scanning " << scan_count(options) << " seeds.\n";
#endif
        }
        std::cout << "Using the " << backend_label(backend)
                  << " scanner.\n";
        if (complete_permutation) {
            // A fully resolved shape order retains the inverse-shuffle fast
            // path. Radius/cage observations are already intersected into the
            // same masks and thus validated without altering the algorithm.
            const auto candidates = scan(
                options, derive_draws(options.observations), backend);
            print_and_save_candidates(candidates, "pillar-scan");
        } else {
            std::cout << "Using exact partial pillar constraints; the complete "
                         "shape shuffle is evaluated for every seed.\n";
            const auto candidates = scan(options, options.observations, backend);
            print_and_save_candidates(candidates, "pillar-scan");
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\nUse --help for usage.\n";
        return EXIT_FAILURE;
    }
}
