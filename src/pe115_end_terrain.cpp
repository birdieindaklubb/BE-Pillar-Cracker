#include "pe115_end_terrain.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace pe115::end_terrain {
namespace {

// Keep ARM VFP operation boundaries explicit.  The target is built with FP
// contraction disabled: `mul_add` is an ordinary multiply followed by add,
// rather than a host fused multiply-add.
namespace fp {
[[nodiscard]] inline float add(float a, float b) noexcept { return a + b; }
[[nodiscard]] inline float sub(float a, float b) noexcept { return a - b; }
[[nodiscard]] inline float mul(float a, float b) noexcept { return a * b; }
[[nodiscard]] inline float div(float a, float b) noexcept { return a / b; }
[[nodiscard]] inline float mul_add(float accumulator, float a, float b) noexcept {
    return add(accumulator, mul(a, b));
}
[[nodiscard]] inline std::int32_t trunc_to_i32(float value) noexcept {
    if (std::isnan(value)) {
        return 0;
    }
    constexpr float positive_limit = 2'147'483'648.0F;
    constexpr float negative_limit = -2'147'483'648.0F;
    if (value >= positive_limit) {
        return std::numeric_limits<std::int32_t>::max();
    }
    if (value <= negative_limit) {
        return std::numeric_limits<std::int32_t>::min();
    }
    return static_cast<std::int32_t>(value);
}
[[nodiscard]] inline std::int32_t floor_to_i32(float value) noexcept {
    std::int32_t integer = trunc_to_i32(value);
    if (value < static_cast<float>(integer)) {
        if (integer == std::numeric_limits<std::int32_t>::min()) {
            return std::numeric_limits<std::int32_t>::max();
        }
        --integer;
    }
    return integer;
}
} // namespace fp

[[nodiscard]] constexpr std::uint32_t bits(std::int32_t value) noexcept {
    return std::bit_cast<std::uint32_t>(value);
}
[[nodiscard]] constexpr std::int32_t signed_bits(std::uint32_t value) noexcept {
    return std::bit_cast<std::int32_t>(value);
}
[[nodiscard]] constexpr std::int32_t wrapping_add(
    std::int32_t left, std::int32_t right) noexcept {
    return signed_bits(bits(left) + bits(right));
}
[[nodiscard]] constexpr std::int32_t wrapping_mul(
    std::int32_t left, std::int32_t right) noexcept {
    return signed_bits(bits(left) * bits(right));
}

class Mt19937 final {
public:
    explicit Mt19937(std::uint32_t seed) noexcept { this->seed(seed); }

    [[nodiscard]] std::uint32_t next_u32() noexcept {
        if (index_ >= state_.size()) {
            twist();
        }
        std::uint32_t value = state_[index_++];
        value ^= value >> 11U;
        value ^= (value << 7U) & 0x9d2c'5680U;
        value ^= (value << 15U) & 0xefc6'0000U;
        value ^= value >> 18U;
        return value;
    }

    [[nodiscard]] float next_float() noexcept {
        return static_cast<float>(static_cast<double>(next_u32()) * 0x1p-32);
    }

    [[nodiscard]] std::uint32_t next_bounded(std::uint32_t bound) noexcept {
        return next_u32() % bound;
    }

private:
    void seed(std::uint32_t value) noexcept {
        state_[0] = value;
        for (std::uint32_t index = 1U; index < state_.size(); ++index) {
            const std::uint32_t previous = state_[index - 1U];
            state_[index] = 1'812'433'253U * (previous ^ (previous >> 30U))
                + index;
        }
        index_ = state_.size();
    }

    void twist() noexcept {
        constexpr std::uint32_t upper_mask = 0x8000'0000U;
        constexpr std::uint32_t lower_mask = 0x7fff'ffffU;
        constexpr std::uint32_t matrix_a = 0x9908'b0dfU;
        constexpr std::size_t period_offset = 397U;
        for (std::size_t index = 0; index < state_.size() - period_offset; ++index) {
            const std::uint32_t joined = (state_[index] & upper_mask)
                | (state_[index + 1U] & lower_mask);
            state_[index] = state_[index + period_offset] ^ (joined >> 1U)
                ^ ((joined & 1U) == 0U ? 0U : matrix_a);
        }
        for (std::size_t index = state_.size() - period_offset;
             index < state_.size() - 1U; ++index) {
            const std::uint32_t joined = (state_[index] & upper_mask)
                | (state_[index + 1U] & lower_mask);
            state_[index] = state_[index + period_offset - state_.size()]
                ^ (joined >> 1U) ^ ((joined & 1U) == 0U ? 0U : matrix_a);
        }
        const std::uint32_t joined = (state_.back() & upper_mask)
            | (state_[0] & lower_mask);
        state_.back() = state_[period_offset - 1U] ^ (joined >> 1U)
            ^ ((joined & 1U) == 0U ? 0U : matrix_a);
        index_ = 0U;
    }

    std::array<std::uint32_t, 624> state_{};
    std::size_t index_{};
};

[[nodiscard]] constexpr std::uint32_t wrap_256(std::int32_t value) noexcept {
    return static_cast<std::uint32_t>(value) & 0xffU;
}

class ImprovedNoise final {
public:
    explicit ImprovedNoise(Mt19937& random) noexcept {
        x_offset_ = fp::mul(random.next_float(), 256.0F);
        y_offset_ = fp::mul(random.next_float(), 256.0F);
        z_offset_ = fp::mul(random.next_float(), 256.0F);
        for (std::uint32_t index = 0; index < 256U; ++index) {
            permutation_[index] = index;
        }
        for (std::uint32_t index = 0; index < 256U; ++index) {
            const std::uint32_t selected = index + random.next_bounded(256U - index);
            std::swap(permutation_[index], permutation_[selected]);
            permutation_[index + 256U] = permutation_[index];
        }
    }

    void add(std::span<float> output, float x, float y, float z,
             std::int32_t x_size, std::int32_t y_size, std::int32_t z_size,
             float x_scale, float y_scale, float z_scale,
             float amplitude) const noexcept {
        const float inverse_amplitude = fp::div(1.0F, amplitude);
        std::size_t output_index = 0U;
        for (std::int32_t ix = 0; ix < x_size; ++ix) {
            const float grid_x = fp::add(x, static_cast<float>(ix));
            const float shifted_x = fp::mul_add(x_offset_, grid_x, x_scale);
            const std::int32_t floor_x = fp::floor_to_i32(shifted_x);
            const std::uint32_t x_index = wrap_256(floor_x);
            const float local_x = fp::sub(shifted_x, static_cast<float>(floor_x));
            const float u = fade(local_x);

            for (std::int32_t iz = 0; iz < z_size; ++iz) {
                const float grid_z = fp::add(z, static_cast<float>(iz));
                const float shifted_z = fp::mul_add(z_offset_, grid_z, z_scale);
                const std::int32_t floor_z = fp::floor_to_i32(shifted_z);
                const std::uint32_t z_index = wrap_256(floor_z);
                const float local_z = fp::sub(shifted_z, static_cast<float>(floor_z));
                const float w = fade(local_z);
                std::int32_t previous_y_index = -1;
                float low_y = 0.0F;
                float high_y = 0.0F;
                float low_y_next_z = 0.0F;
                float high_y_next_z = 0.0F;

                for (std::int32_t iy = 0; iy < y_size; ++iy) {
                    const float grid_y = fp::add(y, static_cast<float>(iy));
                    const float shifted_y = fp::mul_add(y_offset_, grid_y, y_scale);
                    const std::int32_t floor_y = fp::floor_to_i32(shifted_y);
                    const std::uint32_t y_index = wrap_256(floor_y);
                    const float local_y = fp::sub(shifted_y, static_cast<float>(floor_y));
                    const float v = fade(local_y);
                    if (iy == 0 || static_cast<std::int32_t>(y_index) != previous_y_index) {
                        const std::uint32_t a = permutation_[x_index] + y_index;
                        const std::uint32_t aa = permutation_[a] + z_index;
                        const std::uint32_t ab = permutation_[a + 1U] + z_index;
                        const std::uint32_t b = permutation_[x_index + 1U] + y_index;
                        const std::uint32_t ba = permutation_[b] + z_index;
                        const std::uint32_t bb = permutation_[b + 1U] + z_index;
                        const float x_minus_one = fp::sub(local_x, 1.0F);
                        const float y_minus_one = fp::sub(local_y, 1.0F);
                        const float z_minus_one = fp::sub(local_z, 1.0F);
                        low_y = lerp(u,
                            grad(permutation_[aa], local_x, local_y, local_z),
                            grad(permutation_[ba], x_minus_one, local_y, local_z));
                        high_y = lerp(u,
                            grad(permutation_[ab], local_x, y_minus_one, local_z),
                            grad(permutation_[bb], x_minus_one, y_minus_one, local_z));
                        low_y_next_z = lerp(u,
                            grad(permutation_[aa + 1U], local_x, local_y, z_minus_one),
                            grad(permutation_[ba + 1U], x_minus_one, local_y, z_minus_one));
                        high_y_next_z = lerp(u,
                            grad(permutation_[ab + 1U], local_x, y_minus_one, z_minus_one),
                            grad(permutation_[bb + 1U], x_minus_one, y_minus_one, z_minus_one));
                        previous_y_index = static_cast<std::int32_t>(y_index);
                    }
                    const float low_z = lerp(v, low_y, high_y);
                    const float high_z = lerp(v, low_y_next_z, high_y_next_z);
                    const float sample = lerp(w, low_z, high_z);
                    output[output_index] = fp::mul_add(
                        output[output_index], sample, inverse_amplitude);
                    ++output_index;
                }
            }
        }
    }

private:
    [[nodiscard]] static float fade(float value) noexcept {
        const float square = fp::mul(value, value);
        const float cube = fp::mul(square, value);
        const float six_value = fp::mul(value, 6.0F);
        const float inner = fp::sub(six_value, 15.0F);
        return fp::mul(cube, fp::mul_add(10.0F, value, inner));
    }
    [[nodiscard]] static float lerp(float amount, float a, float b) noexcept {
        return fp::mul_add(a, amount, fp::sub(b, a));
    }
    [[nodiscard]] static float grad(std::uint32_t hash, float x, float y, float z) noexcept {
        const std::uint32_t h = hash & 15U;
        const float u = h < 8U ? x : y;
        const float v = h < 4U ? y : ((h == 12U || h == 14U) ? x : z);
        return fp::add((h & 1U) == 0U ? u : -u, (h & 2U) == 0U ? v : -v);
    }

    float x_offset_{};
    float y_offset_{};
    float z_offset_{};
    std::array<std::uint32_t, 512> permutation_{};
};

class PerlinNoise final {
public:
    PerlinNoise(Mt19937& random, std::int32_t octaves) {
        octaves_.reserve(static_cast<std::size_t>(octaves));
        for (std::int32_t octave = 0; octave < octaves; ++octave) {
            octaves_.emplace_back(random);
        }
    }

    void region(std::span<float> output, float x, float y, float z,
                std::int32_t x_size, std::int32_t y_size, std::int32_t z_size,
                float x_scale, float y_scale, float z_scale) const noexcept {
        std::fill(output.begin(), output.end(), 0.0F);
        float octave_scale = 1.0F;
        for (const ImprovedNoise& octave : octaves_) {
            octave.add(output, x, y, z, x_size, y_size, z_size,
                fp::mul(x_scale, octave_scale),
                fp::mul(y_scale, octave_scale),
                fp::mul(z_scale, octave_scale), octave_scale);
            octave_scale = fp::mul(octave_scale, 0.5F);
        }
    }

private:
    std::vector<ImprovedNoise> octaves_;
};

class Simplex2d final {
public:
    explicit Simplex2d(Mt19937& random) noexcept {
        x_offset_ = fp::mul(random.next_float(), 256.0F);
        y_offset_ = fp::mul(random.next_float(), 256.0F);
        z_offset_ = fp::mul(random.next_float(), 256.0F);
        for (std::uint32_t index = 0; index < 256U; ++index) {
            permutation_[index] = index;
        }
        for (std::uint32_t index = 0; index < 256U; ++index) {
            const std::uint32_t selected = index + random.next_bounded(256U - index);
            std::swap(permutation_[index], permutation_[selected]);
            permutation_[index + 256U] = permutation_[index];
        }
    }

    [[nodiscard]] float value(float x, float z) const noexcept {
        constexpr float sqrt3 = std::bit_cast<float>(0x3fddb3d7U);
        const float skew_factor = fp::sub(fp::mul(sqrt3, 0.5F), 0.5F);
        const float skew = fp::mul(skew_factor, fp::add(x, z));
        const std::int32_t cell_x = simplex_floor(fp::add(x, skew));
        const std::int32_t cell_z = simplex_floor(fp::add(z, skew));
        const float unskew_factor = fp::mul(fp::sub(3.0F, sqrt3), 0.16666667F);
        const float unskew = fp::mul(
            static_cast<float>(wrapping_add(cell_x, cell_z)), unskew_factor);
        const float local_x = fp::sub(x, fp::sub(static_cast<float>(cell_x), unskew));
        const float local_z = fp::sub(z, fp::sub(static_cast<float>(cell_z), unskew));
        const std::int32_t offset_x = local_x > local_z ? 1 : 0;
        const std::int32_t offset_z = local_x > local_z ? 0 : 1;
        const float local_x_1 = fp::sub(
            fp::add(local_x, unskew_factor), static_cast<float>(offset_x));
        const float local_z_1 = fp::sub(
            fp::add(local_z, unskew_factor), static_cast<float>(offset_z));
        const float final_offset = fp::sub(
            fp::add(unskew_factor, unskew_factor), 1.0F);
        const float local_x_2 = fp::add(local_x, final_offset);
        const float local_z_2 = fp::add(local_z, final_offset);
        const std::uint32_t px = wrap_256(cell_x);
        const std::uint32_t pz = wrap_256(cell_z);
        const std::uint32_t hash0 = permutation_[px + permutation_[pz]];
        const std::uint32_t hash1 = permutation_[px + static_cast<std::uint32_t>(offset_x)
            + permutation_[pz + static_cast<std::uint32_t>(offset_z)]];
        const std::uint32_t hash2 = permutation_[px + 1U + permutation_[pz + 1U]];
        const float sample0 = contribution(local_x, local_z, gradient(hash0, local_x, local_z));
        const float sample1 = contribution(local_x_1, local_z_1, gradient(hash1, local_x_1, local_z_1));
        const float sample2 = contribution(local_x_2, local_z_2, gradient(hash2, local_x_2, local_z_2));
        return fp::mul(fp::add(fp::add(sample1, sample0), sample2), 70.0F);
    }

private:
    [[nodiscard]] static std::int32_t simplex_floor(float value) noexcept {
        const std::int32_t truncated = fp::trunc_to_i32(value);
        return value <= 0.0F ? signed_bits(bits(truncated) - 1U) : truncated;
    }
    [[nodiscard]] static float contribution(float x, float z, float gradient) noexcept {
        float squared_distance = fp::mul(z, z);
        squared_distance = fp::mul_add(squared_distance, x, x);
        const float attenuation = fp::sub(0.5F, squared_distance);
        if (attenuation < 0.0F) {
            return 0.0F;
        }
        const float squared = fp::mul(attenuation, attenuation);
        return fp::mul(fp::mul(squared, squared), gradient);
    }
    [[nodiscard]] static float gradient(std::uint32_t hash, float x, float z) noexcept {
        constexpr std::array<std::array<std::int32_t, 3>, 12> gradients{{
            {{1, 1, 0}}, {{-1, 1, 0}}, {{1, -1, 0}}, {{-1, -1, 0}},
            {{1, 0, 1}}, {{-1, 0, 1}}, {{1, 0, -1}}, {{-1, 0, -1}},
            {{0, 1, 1}}, {{0, -1, 1}}, {{0, 1, -1}}, {{0, -1, -1}},
        }};
        const auto& g = gradients[hash % 12U];
        return fp::mul_add(fp::mul(x, static_cast<float>(g[0])),
            z, static_cast<float>(g[1]));
    }

    float x_offset_{};
    float y_offset_{}; // Constructed by the native class, unused by value(x, z).
    float z_offset_{};
    std::array<std::uint32_t, 512> permutation_{};
};

[[nodiscard]] constexpr std::size_t lattice_index(
    std::int32_t x, std::int32_t z, std::int32_t y) noexcept {
    return (static_cast<std::size_t>(x) * 3U + static_cast<std::size_t>(z))
        * 33U + static_cast<std::size_t>(y);
}
[[nodiscard]] constexpr std::uint32_t absolute_bits(std::int32_t value) noexcept {
    const std::uint32_t value_bits = bits(value);
    return value < 0 ? 0U - value_bits : value_bits;
}
[[nodiscard]] float distance(float x, float z) noexcept {
    return std::sqrt(fp::add(fp::mul(x, x), fp::mul(z, z)));
}
[[nodiscard]] float clamp_island_height(float value) noexcept {
    float result = -100.0F;
    if (-100.0F < value) {
        result = value;
    }
    if (80.0F < value) {
        result = 80.0F;
    }
    return result;
}

} // namespace

class Generator::Impl final {
public:
    explicit Impl(std::uint32_t world_seed)
        : random_(world_seed), lower_(random_, 16), upper_(random_, 16),
          selector_(random_, 8), island_noise_(random_) {
    }

    [[nodiscard]] std::array<std::uint8_t, Generator::block_count> generate(
        std::int32_t chunk_x, std::int32_t chunk_z) const {
        std::array<float, 297> density{};
        lattice(density, wrapping_mul(chunk_x, 2), wrapping_mul(chunk_z, 2));
        std::array<std::uint8_t, Generator::block_count> blocks{};
        for (std::int32_t cell_x = 0; cell_x < 2; ++cell_x) {
            for (std::int32_t cell_z = 0; cell_z < 2; ++cell_z) {
                for (std::int32_t cell_y = 0; cell_y < 32; ++cell_y) {
                    float z1_x0 = density[lattice_index(cell_x, cell_z + 1, cell_y)];
                    float z1_x1 = density[lattice_index(cell_x + 1, cell_z + 1, cell_y)];
                    float z0_x0 = density[lattice_index(cell_x, cell_z, cell_y)];
                    float z0_x1 = density[lattice_index(cell_x + 1, cell_z, cell_y)];
                    const float z1_x0_y_step = fp::mul(fp::sub(
                        density[lattice_index(cell_x, cell_z + 1, cell_y + 1)], z1_x0), 0.25F);
                    const float z1_x1_y_step = fp::mul(fp::sub(
                        density[lattice_index(cell_x + 1, cell_z + 1, cell_y + 1)], z1_x1), 0.25F);
                    const float z1_x_step = fp::mul(fp::sub(z1_x1, z1_x0), 0.125F);
                    const float z0_x0_y_step = fp::mul(fp::sub(
                        density[lattice_index(cell_x, cell_z, cell_y + 1)], z0_x0), 0.25F);
                    const float z0_x1_y_step = fp::mul(fp::sub(
                        density[lattice_index(cell_x + 1, cell_z, cell_y + 1)], z0_x1), 0.25F);
                    const float z0_x_step = fp::mul(fp::sub(z0_x1, z0_x0), 0.125F);
                    for (std::int32_t local_y = 0; local_y < 4; ++local_y) {
                        float z1 = z1_x0;
                        float z0 = z0_x0;
                        const std::int32_t y = cell_y * 4 + local_y;
                        for (std::int32_t local_x = 0; local_x < 8; ++local_x) {
                            const float z_step = fp::mul(fp::sub(z1, z0), 0.125F);
                            const std::int32_t x = cell_x * 8 + local_x;
                            float value = z0;
                            for (std::int32_t local_z = 0; local_z < 8; ++local_z) {
                                if (value > 0.0F) {
                                    blocks[(static_cast<std::size_t>(x) * 16U
                                        + static_cast<std::size_t>(cell_z * 8 + local_z))
                                        * 128U + static_cast<std::size_t>(y)] = 121U;
                                }
                                if (local_z != 7) {
                                    value = fp::add(value, z_step);
                                }
                            }
                            z1 = fp::add(z1, z1_x_step);
                            z0 = fp::add(z0, z0_x_step);
                        }
                        z1_x1 = fp::add(z1_x1, z1_x1_y_step);
                        z0_x1 = fp::add(z0_x1, z0_x1_y_step);
                        z1_x0 = fp::add(z1_x0, z1_x0_y_step);
                        z0_x0 = fp::add(z0_x0, z0_x0_y_step);
                    }
                }
            }
        }
        return blocks;
    }

private:
    void lattice(std::span<float, 297> output,
                 std::int32_t coarse_x, std::int32_t coarse_z) const {
        std::array<float, 297> selector{};
        std::array<float, 297> lower{};
        std::array<float, 297> upper{};
        selector_.region(selector, static_cast<float>(coarse_x), 0.0F,
            static_cast<float>(coarse_z), 3, 33, 3, 17.1103F, 4.277575F, 17.1103F);
        lower_.region(lower, static_cast<float>(coarse_x), 0.0F,
            static_cast<float>(coarse_z), 3, 33, 3, 1368.824F, 684.412F, 1368.824F);
        upper_.region(upper, static_cast<float>(coarse_x), 0.0F,
            static_cast<float>(coarse_z), 3, 33, 3, 1368.824F, 684.412F, 1368.824F);
        for (std::int32_t lattice_x = 0; lattice_x < 3; ++lattice_x) {
            for (std::int32_t lattice_z = 0; lattice_z < 3; ++lattice_z) {
                const float island_height = island_height_value(
                    coarse_x / 2, coarse_z / 2, lattice_x, lattice_z);
                const std::size_t base = lattice_index(lattice_x, lattice_z, 0);
                std::int32_t vertical = -14;
                std::int32_t bottom_fade = 8;
                for (std::int32_t y = 0; y < 33; ++y) {
                    const std::size_t index = base + static_cast<std::size_t>(y);
                    const float blend = fp::mul_add(0.5F, selector[index], 0.05F);
                    float density = fp::mul(lower[index], 0.001953125F);
                    if (blend >= 0.0F) {
                        if (blend < 1.0F) {
                            const float delta = fp::mul(fp::sub(upper[index], lower[index]), 0.001953125F);
                            density = fp::mul_add(density, delta, blend);
                        } else {
                            density = fp::mul(upper[index], 0.001953125F);
                        }
                    }
                    density = fp::add(fp::sub(island_height, 8.0F), density);
                    if (vertical + 14 < 15) {
                        if (vertical + 14 < 8) {
                            const float fade = static_cast<float>(bottom_fade);
                            const float multiplier = fp::mul_add(1.0F, fade, -0.14285715F);
                            density = fp::mul_add(fp::mul(density, multiplier), fade, -4.2857146F);
                        }
                    } else {
                        float fade = fp::mul(static_cast<float>(vertical), 0.015625F);
                        if (fade < 0.0F) fade = 0.0F;
                        if (fade > 1.0F) fade = 1.0F;
                        density = fp::mul_add(fp::mul(density, fp::sub(1.0F, fade)), fade, -3000.0F);
                    }
                    output[index] = density;
                    ++vertical;
                    --bottom_fade;
                }
            }
        }
    }

    [[nodiscard]] float island_height_value(std::int32_t chunk_x, std::int32_t chunk_z,
                                              std::int32_t local_x, std::int32_t local_z) const noexcept {
        const std::int32_t base_x = wrapping_add(wrapping_mul(chunk_x, 2), local_x);
        const std::int32_t base_z = wrapping_add(wrapping_mul(chunk_z, 2), local_z);
        float result = clamp_island_height(fp::add(100.0F, fp::mul(
            distance(static_cast<float>(base_x), static_cast<float>(base_z)), -8.0F)));
        for (std::int32_t offset_x = -12; offset_x <= 12; ++offset_x) {
            const std::int32_t candidate_x = wrapping_add(chunk_x, offset_x);
            const std::uint32_t abs_x = absolute_bits(candidate_x);
            const std::uint64_t seed_x = static_cast<std::uint64_t>(abs_x) * 3439U;
            const std::int32_t local_delta_x = wrapping_add(local_x, wrapping_mul(offset_x, -2));
            for (std::int32_t offset_z = -12; offset_z <= 12; ++offset_z) {
                const std::int32_t candidate_z = wrapping_add(chunk_z, offset_z);
                const std::uint64_t squared_distance = static_cast<std::uint64_t>(abs_x) * abs_x
                    + static_cast<std::uint64_t>(absolute_bits(candidate_z)) * absolute_bits(candidate_z);
                if (squared_distance <= 4096U
                    || island_noise_.value(static_cast<float>(candidate_x),
                        static_cast<float>(candidate_z)) >= -0.9F) {
                    continue;
                }
                const std::uint64_t island_seed = seed_x
                    + static_cast<std::uint64_t>(absolute_bits(candidate_z)) * 147U;
                const float radius = static_cast<float>(island_seed % 13U + 9U);
                const std::int32_t local_delta_z = wrapping_add(local_z, wrapping_mul(offset_z, -2));
                const float candidate = fp::sub(100.0F, fp::mul(radius,
                    distance(static_cast<float>(local_delta_x), static_cast<float>(local_delta_z))));
                const float clamped = clamp_island_height(candidate);
                if (result < clamped) {
                    result = clamped;
                }
            }
        }
        return result;
    }

    Mt19937 random_;
    PerlinNoise lower_;
    PerlinNoise upper_;
    PerlinNoise selector_;
    Simplex2d island_noise_;
};

Generator::Generator(std::uint32_t world_seed)
    : impl_(std::make_unique<Impl>(world_seed)) {
}
Generator::~Generator() = default;
Generator::Generator(Generator&&) noexcept = default;
Generator& Generator::operator=(Generator&&) noexcept = default;

std::array<std::uint8_t, Generator::block_count> Generator::generate_chunk(
    std::int32_t chunk_x, std::int32_t chunk_z) const {
    return impl_->generate(chunk_x, chunk_z);
}

bool self_test() noexcept {
    struct TestCase final {
        std::uint32_t seed{};
        std::int32_t chunk_x{};
        std::int32_t chunk_z{};
        std::uint64_t expected_hash{};
    };
    constexpr std::array<TestCase, 4> cases{{
        {0U, 0, 0, 0xa2385948afca824dULL},
        {330675023U, 1, 0, 0x1859918b9cbc084cULL},
        {0x9733fc86U, -1, 1, 0x9c0fae6ebca324f3ULL},
        {0xffff'ffffU, 6, -2, 0x0e57ecc23f2e9d73ULL},
    }};
    try {
        for (const TestCase& test : cases) {
            const Generator generator(test.seed);
            const auto blocks = generator.generate_chunk(test.chunk_x, test.chunk_z);
            std::uint64_t hash = 14'695'981'039'346'656'037ULL;
            for (const std::uint8_t block : blocks) {
                hash ^= block;
                hash *= 1'099'511'628'211ULL;
            }
            if (hash != test.expected_hash) {
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

} // namespace pe115::end_terrain
