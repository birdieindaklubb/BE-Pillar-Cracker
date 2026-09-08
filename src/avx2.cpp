// AVX2 implementation of the exact PE 1.1.5 pillar-shuffle predicate.
// This is data parallelism across eight independent candidate seeds; no
// randomness, arithmetic, or candidate is approximated or discarded.

#include "avx2.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace pe115::avx2 {
namespace {

constexpr std::uint32_t kMtMultiplier = 1'812'433'253U;
constexpr std::uint32_t kMtUpperMask = 0x8000'0000U;
constexpr std::uint32_t kMtLowerMask = 0x7fff'ffffU;
constexpr std::uint32_t kMtMatrixA = 0x9908'b0dfU;
constexpr std::size_t kLaneCount = 8;

[[nodiscard]] __m256i temper(__m256i value) noexcept {
    value = _mm256_xor_si256(value, _mm256_srli_epi32(value, 11));
    value = _mm256_xor_si256(value, _mm256_and_si256(
        _mm256_slli_epi32(value, 7),
        _mm256_set1_epi32(static_cast<std::int32_t>(0x9d2c'5680U))));
    value = _mm256_xor_si256(value, _mm256_and_si256(
        _mm256_slli_epi32(value, 15),
        _mm256_set1_epi32(static_cast<std::int32_t>(0xefc6'0000U))));
    return _mm256_xor_si256(value, _mm256_srli_epi32(value, 18));
}

[[nodiscard]] bool residue_matches(
    std::uint32_t value,
    std::size_t draw_index,
    std::uint8_t expected) noexcept {
    switch (draw_index) {
    case 0:
        return (value & 1U) == expected;
    case 1:
        return value % 3U == expected;
    case 2:
        return (value & 3U) == expected;
    case 3:
        return value % 5U == expected;
    case 4:
        return value % 6U == expected;
    case 5:
        return value % 7U == expected;
    case 6:
        return (value & 7U) == expected;
    case 7:
        return value % 9U == expected;
    case 8:
        return value % 10U == expected;
    default:
        return false;
    }
}

[[nodiscard]] std::uint32_t seed_at(
    const ScanSpec& specification,
    std::uint64_t index) noexcept {
    switch (specification.kind) {
    case ScanKind::all:
        return static_cast<std::uint32_t>(index);
    case ScanKind::contiguous_range:
        return static_cast<std::uint32_t>(specification.range_start + index);
    case ScanKind::high_word:
        return (static_cast<std::uint32_t>(specification.word) << 16U)
            | static_cast<std::uint32_t>(index);
    case ScanKind::low_word:
        return (static_cast<std::uint32_t>(index) << 16U) | specification.word;
    }
    return 0U;
}

[[nodiscard]] __m256i seed_step(__m256i value, std::uint32_t index) noexcept {
    const __m256i mixed = _mm256_xor_si256(value, _mm256_srli_epi32(value, 30));
    return _mm256_add_epi32(
        _mm256_mullo_epi32(mixed, _mm256_set1_epi32(
            static_cast<std::int32_t>(kMtMultiplier))),
        _mm256_set1_epi32(static_cast<std::int32_t>(index)));
}

// The first nine MT outputs after initialization use only initial state words
// [0..9] and [397..405].  Retaining those vectors produces precisely the same
// values as a complete 624-word twist.
[[nodiscard]] std::uint8_t matches_eight(
    const std::array<std::uint32_t, kLaneCount>& seeds,
    const std::array<std::uint8_t, 9>& draws) noexcept {
    const __m256i seed_values = _mm256_setr_epi32(
        static_cast<std::int32_t>(seeds[0]), static_cast<std::int32_t>(seeds[1]),
        static_cast<std::int32_t>(seeds[2]), static_cast<std::int32_t>(seeds[3]),
        static_cast<std::int32_t>(seeds[4]), static_cast<std::int32_t>(seeds[5]),
        static_cast<std::int32_t>(seeds[6]), static_cast<std::int32_t>(seeds[7]));

    std::array<__m256i, 10> initial_low{};
    initial_low[0] = seed_values;
    __m256i high_state = seed_values;
    for (std::uint32_t index = 1; index <= 397U; ++index) {
        high_state = seed_step(high_state, index);
        if (index < initial_low.size()) {
            initial_low[index] = high_state;
        }
    }

    const __m256i upper_mask = _mm256_set1_epi32(
        static_cast<std::int32_t>(kMtUpperMask));
    const __m256i lower_mask = _mm256_set1_epi32(
        static_cast<std::int32_t>(kMtLowerMask));
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i matrix_a = _mm256_set1_epi32(
        static_cast<std::int32_t>(kMtMatrixA));
    std::uint8_t viable = 0xffU;

    for (std::size_t index = 0; index < draws.size(); ++index) {
        const __m256i joined = _mm256_or_si256(
            _mm256_and_si256(initial_low[index], upper_mask),
            _mm256_and_si256(initial_low[index + 1U], lower_mask));
        const __m256i odd_mask = _mm256_cmpeq_epi32(
            _mm256_and_si256(joined, one), one);
        const __m256i twisted = _mm256_xor_si256(
            _mm256_xor_si256(high_state, _mm256_srli_epi32(joined, 1)),
            _mm256_and_si256(odd_mask, matrix_a));

        alignas(32) std::array<std::uint32_t, kLaneCount> values{};
        _mm256_store_si256(
            reinterpret_cast<__m256i*>(values.data()), temper(twisted));
        for (std::size_t lane = 0; lane < values.size(); ++lane) {
            const std::uint8_t bit = static_cast<std::uint8_t>(1U << lane);
            if ((viable & bit) != 0U
                && !residue_matches(values[lane], index, draws[index])) {
                viable = static_cast<std::uint8_t>(viable & ~bit);
            }
        }
        if (viable == 0U || index + 1U == draws.size()) {
            break;
        }
        high_state = seed_step(
            high_state, static_cast<std::uint32_t>(398U + index));
    }
    return viable;
}

[[nodiscard]] std::uint8_t matches_eight(
    const std::array<std::uint32_t, kLaneCount>& seeds,
    const PillarShapeMasks& allowed_shapes) noexcept {
    const __m256i seed_values = _mm256_setr_epi32(
        static_cast<std::int32_t>(seeds[0]), static_cast<std::int32_t>(seeds[1]),
        static_cast<std::int32_t>(seeds[2]), static_cast<std::int32_t>(seeds[3]),
        static_cast<std::int32_t>(seeds[4]), static_cast<std::int32_t>(seeds[5]),
        static_cast<std::int32_t>(seeds[6]), static_cast<std::int32_t>(seeds[7]));

    std::array<__m256i, 10> initial_low{};
    initial_low[0] = seed_values;
    __m256i high_state = seed_values;
    for (std::uint32_t index = 1; index <= 397U; ++index) {
        high_state = seed_step(high_state, index);
        if (index < initial_low.size()) {
            initial_low[index] = high_state;
        }
    }

    const __m256i upper_mask = _mm256_set1_epi32(
        static_cast<std::int32_t>(kMtUpperMask));
    const __m256i lower_mask = _mm256_set1_epi32(
        static_cast<std::int32_t>(kMtLowerMask));
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i matrix_a = _mm256_set1_epi32(
        static_cast<std::int32_t>(kMtMatrixA));
    alignas(32) std::array<std::array<std::uint32_t, kLaneCount>, 9>
        random_values{};

    for (std::size_t index = 0; index < random_values.size(); ++index) {
        const __m256i joined = _mm256_or_si256(
            _mm256_and_si256(initial_low[index], upper_mask),
            _mm256_and_si256(initial_low[index + 1U], lower_mask));
        const __m256i odd_mask = _mm256_cmpeq_epi32(
            _mm256_and_si256(joined, one), one);
        const __m256i twisted = _mm256_xor_si256(
            _mm256_xor_si256(high_state, _mm256_srli_epi32(joined, 1)),
            _mm256_and_si256(odd_mask, matrix_a));
        _mm256_store_si256(reinterpret_cast<__m256i*>(
            random_values[index].data()), temper(twisted));
        if (index + 1U != random_values.size()) {
            high_state = seed_step(
                high_state, static_cast<std::uint32_t>(398U + index));
        }
    }

    std::uint8_t viable = 0U;
    for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
        std::array<std::uint8_t, kPillarCount> order{};
        for (std::uint8_t shape = 0U; shape < kPillarCount; ++shape) {
            order[shape] = shape;
        }
        for (std::size_t index = 1; index < order.size(); ++index) {
            const std::size_t selected = random_values[index - 1U][lane]
                % static_cast<std::uint32_t>(index + 1U);
            const std::uint8_t replacement = order[selected];
            order[selected] = order[index];
            order[index] = replacement;
        }
        bool matches = true;
        for (std::size_t index = 0; index < order.size(); ++index) {
            const std::uint16_t shape_bit = static_cast<std::uint16_t>(
                1U << order[index]);
            if ((allowed_shapes[index] & shape_bit) == 0U) {
                matches = false;
                break;
            }
        }
        if (matches) {
            viable = static_cast<std::uint8_t>(viable | (1U << lane));
        }
    }
    return viable;
}

} // namespace

bool is_available() noexcept {
#if defined(_MSC_VER)
    int registers[4]{};
    __cpuidex(registers, 0, 0);
    if (registers[0] < 7) {
        return false;
    }
    __cpuidex(registers, 1, 0);
    constexpr int kOsXsaveBit = 1 << 27;
    constexpr int kAvxBit = 1 << 28;
    if ((registers[2] & (kOsXsaveBit | kAvxBit))
        != (kOsXsaveBit | kAvxBit)) {
        return false;
    }
    if ((_xgetbv(0) & 0x6U) != 0x6U) {
        return false;
    }
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0;
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

std::vector<std::uint32_t> scan(
    const ScanSpec& specification,
    const std::array<std::uint8_t, 9>& draws) {
    std::vector<std::uint32_t> candidates;
    const auto total = static_cast<long long>(specification.count);

#ifdef _OPENMP
#pragma omp parallel
    {
        std::vector<std::uint32_t> local_candidates;
        local_candidates.reserve(64U);
#pragma omp for schedule(static)
        for (long long base = 0; base < total;
             base += static_cast<long long>(kLaneCount)) {
            std::array<std::uint32_t, kLaneCount> seeds{};
            for (std::size_t lane = 0; lane < seeds.size(); ++lane) {
                seeds[lane] = seed_at(
                    specification, static_cast<std::uint64_t>(base) + lane);
            }
            const std::uint8_t viable = matches_eight(seeds, draws);
            for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
                if ((viable & static_cast<std::uint8_t>(1U << lane)) != 0U
                    && static_cast<std::uint64_t>(base) + lane
                        < specification.count) {
                    local_candidates.push_back(seeds[lane]);
                }
            }
        }
#pragma omp critical
        candidates.insert(
            candidates.end(), local_candidates.begin(), local_candidates.end());
    }
#else
    for (std::uint64_t base = 0; base < specification.count; base += kLaneCount) {
        std::array<std::uint32_t, kLaneCount> seeds{};
        for (std::size_t lane = 0; lane < seeds.size(); ++lane) {
            seeds[lane] = seed_at(specification, base + lane);
        }
        const std::uint8_t viable = matches_eight(seeds, draws);
        for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
            if ((viable & static_cast<std::uint8_t>(1U << lane)) != 0U
                && base + lane < specification.count) {
                candidates.push_back(seeds[lane]);
            }
        }
    }
#endif
    return candidates;
}

std::vector<std::uint32_t> scan(
    const ScanSpec& specification,
    const PillarShapeMasks& allowed_shapes) {
    std::vector<std::uint32_t> candidates;
    const auto total = static_cast<long long>(specification.count);

#ifdef _OPENMP
#pragma omp parallel
    {
        std::vector<std::uint32_t> local_candidates;
        local_candidates.reserve(64U);
#pragma omp for schedule(static)
        for (long long base = 0; base < total;
             base += static_cast<long long>(kLaneCount)) {
            std::array<std::uint32_t, kLaneCount> seeds{};
            for (std::size_t lane = 0; lane < seeds.size(); ++lane) {
                seeds[lane] = seed_at(
                    specification, static_cast<std::uint64_t>(base) + lane);
            }
            const std::uint8_t viable = matches_eight(seeds, allowed_shapes);
            for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
                if ((viable & static_cast<std::uint8_t>(1U << lane)) != 0U
                    && static_cast<std::uint64_t>(base) + lane
                        < specification.count) {
                    local_candidates.push_back(seeds[lane]);
                }
            }
        }
#pragma omp critical
        candidates.insert(
            candidates.end(), local_candidates.begin(), local_candidates.end());
    }
#else
    for (std::uint64_t base = 0; base < specification.count; base += kLaneCount) {
        std::array<std::uint32_t, kLaneCount> seeds{};
        for (std::size_t lane = 0; lane < seeds.size(); ++lane) {
            seeds[lane] = seed_at(specification, base + lane);
        }
        const std::uint8_t viable = matches_eight(seeds, allowed_shapes);
        for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
            if ((viable & static_cast<std::uint8_t>(1U << lane)) != 0U
                && base + lane < specification.count) {
                candidates.push_back(seeds[lane]);
            }
        }
    }
#endif
    return candidates;
}

} // namespace pe115::avx2
