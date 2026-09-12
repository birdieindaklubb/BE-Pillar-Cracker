#include "pillar_layout.hpp"

namespace pe115::pillars {

std::array<std::uint32_t, kPillarCount - 1U> shuffle_words(
    std::uint32_t world_seed) noexcept {
    std::array<std::uint32_t, 10U> initial_low{};
    std::array<std::uint32_t, kPillarCount - 1U> initial_high{};
    initial_low[0] = world_seed;

    std::uint32_t state_word = world_seed;
    for (std::uint32_t index = 1U; index <= 405U; ++index) {
        state_word = seed_step(state_word, index);
        if (index < initial_low.size()) {
            initial_low[index] = state_word;
        }
        if (index >= 397U) {
            initial_high[index - 397U] = state_word;
        }
    }

    std::array<std::uint32_t, kPillarCount - 1U> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        const std::uint32_t joined =
            (initial_low[index] & mt_upper_mask)
            | (initial_low[index + 1U] & mt_lower_mask);
        std::uint32_t twisted = initial_high[index] ^ (joined >> 1U);
        if ((joined & 1U) != 0U) {
            twisted ^= mt_matrix_a;
        }
        result[index] = temper(twisted);
    }
    return result;
}

std::array<std::uint8_t, kPillarCount> shapes(
    std::uint32_t world_seed) noexcept {
    std::array<std::uint8_t, kPillarCount> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = static_cast<std::uint8_t>(index);
    }

    const auto words = shuffle_words(world_seed);
    for (std::size_t index = 1U; index < result.size(); ++index) {
        const std::size_t selected = words[index - 1U]
            % static_cast<std::uint32_t>(index + 1U);
        const std::uint8_t value = result[selected];
        result[selected] = result[index];
        result[index] = value;
    }
    return result;
}

} // namespace pe115::pillars
