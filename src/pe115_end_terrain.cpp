#include "pe115_end_terrain.hpp"
#include "pillar_layout.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
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
[[nodiscard]] constexpr std::int32_t wrapping_sub(
    std::int32_t left, std::int32_t right) noexcept {
    return signed_bits(bits(left) - bits(right));
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
        // PE's Random converts the complete MT output to binary32 before the
        // 2^-32 scaling multiplication.  Do not replace this with a direct
        // double conversion: that shifts native End-noise threshold cells.
        const float rounded_integer = static_cast<float>(next_u32());
        return static_cast<float>(
            static_cast<double>(rounded_integer) * 0x1p-32);
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
                        low_y = blend_x(
                            gradient_terms(
                                permutation_[aa], local_x, local_y, local_z),
                            gradient_terms(
                                permutation_[ba], x_minus_one, local_y, local_z),
                            u);
                        high_y = blend_x(
                            gradient_terms(
                                permutation_[ab], local_x, y_minus_one, local_z),
                            gradient_terms(
                                permutation_[bb],
                                x_minus_one,
                                y_minus_one,
                                local_z),
                            u);
                        low_y_next_z = blend_x(
                            gradient_terms(
                                permutation_[aa + 1U],
                                local_x,
                                local_y,
                                z_minus_one),
                            gradient_terms(
                                permutation_[ba + 1U],
                                x_minus_one,
                                local_y,
                                z_minus_one),
                            u);
                        high_y_next_z = blend_x(
                            gradient_terms(
                                permutation_[ab + 1U],
                                local_x,
                                y_minus_one,
                                z_minus_one),
                            gradient_terms(
                                permutation_[bb + 1U],
                                x_minus_one,
                                y_minus_one,
                                z_minus_one),
                            u);
                        previous_y_index = static_cast<std::int32_t>(y_index);
                    }
                    // Preserve the native ARM evaluation order: resolve the
                    // Z=0 Y blend, then derive the rounded Z delta from it.
                    // A conventional pair of independent Y blends is only
                    // algebraically equivalent and flips threshold blocks.
                    const float y_blend_at_z0 = lerp(v, low_y, high_y);
                    const float z_delta = fp::add(
                        fp::sub(low_y_next_z, y_blend_at_z0),
                        fp::mul(
                            fp::sub(high_y_next_z, low_y_next_z), v));
                    const float sample = fp::mul_add(
                        y_blend_at_z0, w, z_delta);
                    output[output_index] = fp::mul_add(
                        output[output_index], sample, inverse_amplitude);
                    ++output_index;
                }
            }
        }
    }

private:
    struct GradientTerms final {
        float u{};
        float v{};
    };

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
    [[nodiscard]] static GradientTerms gradient_terms(
        std::uint32_t hash,
        float x,
        float y,
        float z) noexcept {
        const std::uint32_t h = hash & 15U;
        const float first = h < 8U ? x : y;
        const float second = h < 4U ? y : ((h == 12U || h == 14U) ? x : z);
        return {
            (h & 1U) == 0U ? first : -first,
            (h & 2U) == 0U ? second : -second,
        };
    }
    [[nodiscard]] static float blend_x(
        GradientTerms current,
        GradientTerms next,
        float amount) noexcept {
        const float current_sum = fp::add(current.u, current.v);
        const float delta = fp::add(
            fp::sub(next.u, current_sum), next.v);
        return fp::mul_add(current_sum, amount, delta);
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
        float attenuation = fp::sub(0.5F, fp::mul(x, x));
        attenuation = fp::sub(attenuation, fp::mul(z, z));
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
        return fp::mul_add(fp::mul(z, static_cast<float>(g[1])),
            x, static_cast<float>(g[0]));
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

constexpr std::uint8_t kAir = 0U;
constexpr std::uint8_t kObsidian = 49U;
constexpr std::uint8_t kTorch = 50U;
constexpr std::uint8_t kFire = 51U;
constexpr std::uint8_t kBedrock = 7U;
constexpr std::uint8_t kIronBars = 101U;

[[nodiscard]] constexpr std::size_t block_index(
    std::int32_t local_x,
    std::int32_t y,
    std::int32_t local_z) noexcept {
    return (static_cast<std::size_t>(local_x) * 16U
        + static_cast<std::size_t>(local_z)) * 128U
        + static_cast<std::size_t>(y);
}

class ChunkOverlay final {
public:
    ChunkOverlay(
        std::array<std::uint8_t, Generator::block_count>& blocks,
        std::int32_t chunk_x,
        std::int32_t chunk_z) noexcept
        : blocks_(blocks),
          origin_x_(wrapping_mul(chunk_x, Generator::chunk_width)),
          origin_z_(wrapping_mul(chunk_z, Generator::chunk_width)) {
    }

    void set(
        std::int32_t x,
        std::int32_t y,
        std::int32_t z,
        std::uint8_t block) noexcept {
        if (y < 0 || y >= Generator::chunk_height) {
            return;
        }
        const std::int32_t local_x = wrapping_sub(x, origin_x_);
        const std::int32_t local_z = wrapping_sub(z, origin_z_);
        if (local_x < 0 || local_x >= Generator::chunk_width
            || local_z < 0 || local_z >= Generator::chunk_width) {
            return;
        }
        blocks_[block_index(local_x, y, local_z)] = block;
    }

    [[nodiscard]] std::uint8_t get(
        std::int32_t x,
        std::int32_t y,
        std::int32_t z) const noexcept {
        if (y < 0 || y >= Generator::chunk_height) {
            return kAir;
        }
        const std::int32_t local_x = wrapping_sub(x, origin_x_);
        const std::int32_t local_z = wrapping_sub(z, origin_z_);
        if (local_x < 0 || local_x >= Generator::chunk_width
            || local_z < 0 || local_z >= Generator::chunk_width) {
            return kAir;
        }
        return blocks_[block_index(local_x, y, local_z)];
    }

private:
    std::array<std::uint8_t, Generator::block_count>& blocks_;
    std::int32_t origin_x_{};
    std::int32_t origin_z_{};
};

void place_pillar(
    ChunkOverlay& blocks,
    pillars::Site site,
    std::uint8_t shape) noexcept {
    const std::int32_t radius = pillars::radius_for_shape(shape);
    const std::int32_t height = pillars::feature_height_for_shape(shape);
    const std::int32_t radius_squared = radius * radius;
    for (std::int32_t x = site.x - radius; x <= site.x + radius; ++x) {
        for (std::int32_t z = site.z - radius; z <= site.z + radius; ++z) {
            const std::int32_t dx = x - site.x;
            const std::int32_t dz = z - site.z;
            const bool inside = dx * dx + dz * dz <= radius_squared;
            for (std::int32_t y = 0; y <= height + 10; ++y) {
                if (inside && y < height) {
                    blocks.set(x, y, z, kObsidian);
                } else if (y > 65) {
                    blocks.set(x, y, z, kAir);
                }
            }
        }
    }

    if (pillars::is_caged_shape(shape)) {
        for (std::int32_t dx = -2; dx <= 2; ++dx) {
            for (std::int32_t dz = -2; dz <= 2; ++dz) {
                const bool outer = dx == -2 || dx == 2 || dz == -2 || dz == 2;
                if (outer) {
                    blocks.set(site.x + dx, height, site.z + dz, kIronBars);
                    blocks.set(site.x + dx, height + 1, site.z + dz, kIronBars);
                    blocks.set(site.x + dx, height + 2, site.z + dz, kIronBars);
                }
                blocks.set(site.x + dx, height + 3, site.z + dz, kIronBars);
            }
        }
    }

    blocks.set(site.x, height, site.z, kBedrock);
    blocks.set(site.x, height + 1, site.z, kFire);
}

void place_inactive_podium(
    ChunkOverlay& blocks,
    std::int32_t center_y) noexcept {
    constexpr float outer_squared = 12.25F;
    constexpr float inner_squared = 6.25F;
    for (std::int32_t dx = -4; dx <= 4; ++dx) {
        for (std::int32_t dy = -1; dy <= 32; ++dy) {
            for (std::int32_t dz = -4; dz <= 4; ++dz) {
                const float distance_squared = static_cast<float>(dx * dx + dz * dz);
                if (distance_squared > outer_squared) {
                    continue;
                }
                if (dy < 0) {
                    blocks.set(dx, center_y + dy, dz,
                        distance_squared <= inner_squared ? kBedrock : 121U);
                } else if (dy == 0) {
                    blocks.set(dx, center_y, dz,
                        distance_squared > inner_squared ? kBedrock : kAir);
                } else {
                    blocks.set(dx, center_y + dy, dz, kAir);
                }
            }
        }
    }
    for (std::int32_t dy = 0; dy < 4; ++dy) {
        blocks.set(0, center_y + dy, 0, kBedrock);
    }
    const std::int32_t torch_y = center_y + 2;
    blocks.set(0, torch_y, -1, kTorch);
    blocks.set(0, torch_y, 1, kTorch);
    blocks.set(1, torch_y, 0, kTorch);
    blocks.set(-1, torch_y, 0, kTorch);
}

void place_arrival_platform(ChunkOverlay& blocks) noexcept {
    for (std::int32_t dx = -2; dx <= 2; ++dx) {
        for (std::int32_t dy = -1; dy <= 2; ++dy) {
            for (std::int32_t dz = -2; dz <= 2; ++dz) {
                blocks.set(100 + dx, 49 + dy, dz,
                    dy == -1 ? kObsidian : kAir);
            }
        }
    }
}

} // namespace

class Generator::Impl final {
public:
    explicit Impl(std::uint32_t world_seed)
        : world_seed_(world_seed), random_(world_seed), lower_(random_, 16), upper_(random_, 16),
          selector_(random_, 8), island_noise_(random_) {
    }

    class BlockWindow final {
    public:
        explicit BlockWindow(const Impl& terrain) noexcept
            : terrain_(terrain) {
        }

        [[nodiscard]] std::uint8_t get(
            std::int32_t x,
            std::int32_t y,
            std::int32_t z) {
            if (y < 0 || y >= Generator::chunk_height) {
                return kAir;
            }
            std::int32_t chunk_x{};
            std::int32_t local_x{};
            std::int32_t chunk_z{};
            std::int32_t local_z{};
            split(x, chunk_x, local_x);
            split(z, chunk_z, local_z);
            return chunk(chunk_x, chunk_z)[block_index(local_x, y, local_z)];
        }

        void set(
            std::int32_t x,
            std::int32_t y,
            std::int32_t z,
            std::uint8_t block) {
            if (y < 0 || y >= Generator::chunk_height) {
                return;
            }
            std::int32_t chunk_x{};
            std::int32_t local_x{};
            std::int32_t chunk_z{};
            std::int32_t local_z{};
            split(x, chunk_x, local_x);
            split(z, chunk_z, local_z);
            chunk(chunk_x, chunk_z)[block_index(local_x, y, local_z)] = block;
        }

        [[nodiscard]] const std::array<std::uint8_t, Generator::block_count>&
        target(std::int32_t chunk_x, std::int32_t chunk_z) {
            return chunk(chunk_x, chunk_z);
        }

    private:
        static void split(
            std::int32_t coordinate,
            std::int32_t& chunk_coordinate,
            std::int32_t& local_coordinate) noexcept {
            chunk_coordinate = coordinate / Generator::chunk_width;
            local_coordinate = coordinate % Generator::chunk_width;
            if (local_coordinate < 0) {
                --chunk_coordinate;
                local_coordinate += Generator::chunk_width;
            }
        }

        [[nodiscard]] std::array<std::uint8_t, Generator::block_count>& chunk(
            std::int32_t chunk_x,
            std::int32_t chunk_z) {
            const auto key = std::pair{chunk_x, chunk_z};
            const auto found = chunks_.find(key);
            if (found != chunks_.end()) {
                return found->second;
            }
            return chunks_.emplace(key, terrain_.generate(chunk_x, chunk_z))
                .first->second;
        }

        const Impl& terrain_;
        std::map<std::pair<std::int32_t, std::int32_t>,
            std::array<std::uint8_t, Generator::block_count>> chunks_;
    };

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
                    const float z0_x0_y_step = fp::mul(fp::sub(
                        density[lattice_index(cell_x, cell_z, cell_y + 1)], z0_x0), 0.25F);
                    const float z0_x1_y_step = fp::mul(fp::sub(
                        density[lattice_index(cell_x + 1, cell_z, cell_y + 1)], z0_x1), 0.25F);
                for (std::int32_t local_y = 0; local_y < 4; ++local_y) {
                    float z1 = z1_x0;
                    float z0 = z0_x0;
                    // The native loop updates the vertical corner values,
                    // then derives fresh horizontal slopes for this layer.
                    // Reusing a single slope across all four layers changes
                    // binary32 threshold blocks.
                    const float z1_x_step = fp::mul(
                        fp::sub(z1_x1, z1_x0), 0.125F);
                    const float z0_x_step = fp::mul(
                        fp::sub(z0_x1, z0_x0), 0.125F);
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

    [[nodiscard]] std::array<std::uint8_t, Generator::block_count>
    generate_fresh_entry(
        std::int32_t chunk_x,
        std::int32_t chunk_z) const {
        auto blocks = generate(chunk_x, chunk_z);
        apply_fresh_scene(blocks, chunk_x, chunk_z);
        return blocks;
    }

    [[nodiscard]] std::array<std::uint8_t, Generator::block_count>
    generate_fresh_visible(
        std::int32_t chunk_x,
        std::int32_t chunk_z) const {
        auto blocks = generate_decorated(chunk_x, chunk_z);
        apply_fresh_scene(blocks, chunk_x, chunk_z);
        return blocks;
    }

    void apply_fresh_scene(
        std::array<std::uint8_t, Generator::block_count>& blocks,
        std::int32_t chunk_x,
        std::int32_t chunk_z) const {
        ChunkOverlay overlay(blocks, chunk_x, chunk_z);

        const auto shapes = pillars::shapes(world_seed_);
        for (std::size_t index = 0; index < pillars::sites.size(); ++index) {
            place_pillar(overlay, pillars::sites[index], shapes[index]);
        }

        // EndDragonFight derives the inactive podium center from the raw
        // terrain height at (0, 0), so it is independent of which central
        // chunk is being queried here.
        const auto center = generate(0, 0);
        std::int32_t center_y = Generator::chunk_height - 1;
        while (center_y > 0
            && center[block_index(0, center_y - 1, 0)] == kAir) {
            --center_y;
        }
        place_inactive_podium(overlay, center_y);
        place_arrival_platform(overlay);
    }

    [[nodiscard]] std::array<std::uint8_t, Generator::block_count>
    generate_decorated(
        std::int32_t chunk_x,
        std::int32_t chunk_z) const {
        BlockWindow window(*this);
        // A natural End decoration can start in this chunk or its west,
        // north, or north-west neighbour.  Their bounded features are the
        // only decorator writes that can reach the requested chunk.
        for (std::int32_t source_z = chunk_z - 1;
             source_z <= chunk_z;
             ++source_z) {
            for (std::int32_t source_x = chunk_x - 1;
                 source_x <= chunk_x;
                 ++source_x) {
                populate_outer_chunk(window, source_x, source_z);
            }
        }
        return window.target(chunk_x, chunk_z);
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
                    // prepareHeights doubles chunk coordinates; getHeights
                    // halves them before delegating to this routine.
                    coarse_x / 2, coarse_z / 2, lattice_x, lattice_z);
                const std::size_t base = lattice_index(lattice_x, lattice_z, 0);
                std::int32_t vertical = -14;
                std::int32_t bottom_fade = 8;
                for (std::int32_t y = 0; y < 33; ++y) {
                    const std::size_t index = base + static_cast<std::size_t>(y);
                    // VFP computes selector * 0.05F + 0.5F.
                    // VFP: selector * 0.05F + 0.5F.  fp::mul_add is
                    // accumulator + a * b, so the offset comes first.
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

    [[nodiscard]] static bool outside_main_island(
        std::int32_t chunk_x,
        std::int32_t chunk_z) noexcept {
        const std::uint32_t distance_squared = bits(chunk_x) * bits(chunk_x)
            + bits(chunk_z) * bits(chunk_z);
        return distance_squared > 4096U;
    }

    [[nodiscard]] Mt19937 population_random(
        std::int32_t chunk_x,
        std::int32_t chunk_z) const noexcept {
        Mt19937 seed_random(world_seed_);
        const std::uint32_t x_multiplier = (seed_random.next_u32() >> 1U) | 1U;
        const std::uint32_t z_multiplier = (seed_random.next_u32() >> 1U) | 1U;
        return Mt19937(world_seed_ ^ (bits(chunk_x) * x_multiplier
            + bits(chunk_z) * z_multiplier));
    }

    static void advance_linear_ore_words(
        Mt19937& random,
        std::uint32_t count,
        std::uint32_t size) noexcept {
        for (std::uint32_t occurrence = 0; occurrence < count; ++occurrence) {
            for (std::uint32_t word = 0; word < 6U + size; ++word) {
                static_cast<void>(random.next_u32());
            }
        }
    }

    static void advance_triangular_ore_words(
        Mt19937& random,
        std::uint32_t count,
        std::uint32_t size) noexcept {
        for (std::uint32_t occurrence = 0; occurrence < count; ++occurrence) {
            for (std::uint32_t word = 0; word < 7U + size; ++word) {
                static_cast<void>(random.next_u32());
            }
        }
    }

    static void advance_end_ore_words(Mt19937& random) noexcept {
        // These inherited decorator features never replace End stone, but
        // their exact draw count determines all subsequent End decorations.
        advance_linear_ore_words(random, 10U, 33U);
        advance_linear_ore_words(random, 8U, 33U);
        if ((random.next_u32() & 15U) == 0U) {
            advance_linear_ore_words(random, 50U, 33U);
        }
        advance_linear_ore_words(random, 10U, 33U); // diorite
        advance_linear_ore_words(random, 10U, 33U); // granite
        advance_linear_ore_words(random, 10U, 33U); // andesite
        advance_linear_ore_words(random, 20U, 17U);
        advance_linear_ore_words(random, 20U, 9U);
        advance_linear_ore_words(random, 2U, 9U);
        advance_linear_ore_words(random, 8U, 8U);
        advance_linear_ore_words(random, 1U, 8U);
        advance_triangular_ore_words(random, 1U, 7U);
    }

    static void place_outer_island(
        BlockWindow& blocks,
        Mt19937& random,
        std::int32_t x,
        std::int32_t y,
        std::int32_t z) noexcept {
        float radius = static_cast<float>(random.next_bounded(3U) | 4U);
        std::int32_t y_offset = 0;
        while (radius > 0.5F) {
            std::int32_t extent = fp::trunc_to_i32(radius);
            if (static_cast<float>(extent) != radius) {
                ++extent;
            }
            const float boundary = fp::mul(
                fp::add(radius, 1.0F), fp::add(radius, 1.0F));
            for (std::int32_t dx = -extent; dx <= extent; ++dx) {
                for (std::int32_t dz = -extent; dz <= extent; ++dz) {
                    if (static_cast<float>(dx * dx + dz * dz) <= boundary) {
                        blocks.set(x + dx, y + y_offset, z + dz, 121U);
                    }
                }
            }
            --y_offset;
            radius = fp::sub(
                fp::sub(radius, 0.5F),
                static_cast<float>(random.next_u32() & 1U));
        }
    }

    static std::int32_t surface_y(
        BlockWindow& blocks,
        std::int32_t x,
        std::int32_t z) {
        for (std::int32_t y = Generator::chunk_height - 1; y >= 0; --y) {
            if (blocks.get(x, y, z) != kAir) {
                return y;
            }
        }
        return -1;
    }

    static bool chorus_neighbors_are_air(
        BlockWindow& blocks,
        std::int32_t x,
        std::int32_t y,
        std::int32_t z,
        std::int32_t ignored_direction = -1) {
        constexpr std::array<std::pair<std::int32_t, std::int32_t>, 4U> steps{{
            {0, -1}, {1, 0}, {0, 1}, {-1, 0},
        }};
        for (std::int32_t direction = 0; direction < 4; ++direction) {
            if (direction == ignored_direction) {
                continue;
            }
            const auto [dx, dz] = steps[static_cast<std::size_t>(direction)];
            if (blocks.get(wrapping_add(x, dx), y, wrapping_add(z, dz)) != kAir) {
                return false;
            }
        }
        return true;
    }

    static void grow_chorus(
        BlockWindow& blocks,
        Mt19937& random,
        std::int32_t x,
        std::int32_t y,
        std::int32_t z,
        std::int32_t root_x,
        std::int32_t root_z,
        std::int32_t spread_limit,
        std::int32_t layer) {
        std::int32_t vertical_length = static_cast<std::int32_t>(
            random.next_u32() & 3U) + 1;
        if (layer == 0) {
            ++vertical_length;
        }

        for (std::int32_t offset = 1; offset <= vertical_length; ++offset) {
            if (!chorus_neighbors_are_air(blocks, x, y + offset, z)) {
                return;
            }
            blocks.set(x, y + offset, z, 199U);
        }

        bool branched = false;
        if (layer < 4) {
            std::int32_t attempts = static_cast<std::int32_t>(random.next_u32() & 3U);
            if (layer == 0) {
                ++attempts;
            }
            constexpr std::array<std::pair<std::int32_t, std::int32_t>, 4U> steps{{
                {0, -1}, {1, 0}, {0, 1}, {-1, 0},
            }};
            for (std::int32_t attempt = 0; attempt < attempts; ++attempt) {
                const std::int32_t direction = static_cast<std::int32_t>(
                    random.next_u32() & 3U);
                const auto [dx, dz] = steps[static_cast<std::size_t>(direction)];
                const std::int32_t branch_x = wrapping_add(x, dx);
                const std::int32_t branch_y = y + vertical_length;
                const std::int32_t branch_z = wrapping_add(z, dz);
                const std::int32_t delta_x = branch_x - root_x;
                const std::int32_t delta_z = branch_z - root_z;
                const std::int32_t opposite = (direction + 2) & 3;
                if (delta_x <= -spread_limit || delta_x >= spread_limit
                    || delta_z <= -spread_limit || delta_z >= spread_limit
                    || blocks.get(branch_x, branch_y, branch_z) != kAir
                    || blocks.get(branch_x, branch_y - 1, branch_z) != kAir
                    || !chorus_neighbors_are_air(
                        blocks, branch_x, branch_y, branch_z, opposite)) {
                    continue;
                }
                branched = true;
                blocks.set(branch_x, branch_y, branch_z, 199U);
                grow_chorus(
                    blocks, random, branch_x, branch_y, branch_z, root_x,
                    root_z, spread_limit, layer + 1);
            }
        }

        if (!branched) {
            blocks.set(x, y + vertical_length, z, 200U);
        }
    }

    static void place_chorus(
        BlockWindow& blocks,
        Mt19937& random,
        std::int32_t x,
        std::int32_t y,
        std::int32_t z) {
        blocks.set(x, y, z, 199U);
        grow_chorus(blocks, random, x, y, z, x, z, 8, 0);
    }

    static void place_gateway(
        BlockWindow& blocks,
        std::int32_t x,
        std::int32_t y,
        std::int32_t z) {
        for (std::int32_t dx = -1; dx <= 1; ++dx) {
            for (std::int32_t dy = -2; dy <= 2; ++dy) {
                for (std::int32_t dz = -1; dz <= 1; ++dz) {
                    const bool central = dx == 0 && dy == 0 && dz == 0;
                    const bool cap = dx == 0 && dz == 0 && (dy == -2 || dy == 2);
                    const bool cross = (dx == 0 || dz == 0)
                        && dy != -2 && dy != 2;
                    blocks.set(x + dx, y + dy, z + dz,
                        central ? 209U : (cap || cross) ? kBedrock : kAir);
                }
            }
        }
    }

    void populate_outer_chunk(
        BlockWindow& blocks,
        std::int32_t chunk_x,
        std::int32_t chunk_z) const {
        Mt19937 random = population_random(chunk_x, chunk_z);
        advance_end_ore_words(random);
        const std::int32_t origin_x = wrapping_mul(chunk_x, 16);
        const std::int32_t origin_z = wrapping_mul(chunk_z, 16);

        if (outside_main_island(chunk_x, chunk_z)
            && island_height_value(chunk_x, chunk_z, 1, 1) < -20.0F
            && random.next_bounded(14U) == 0U) {
            const std::int32_t x = wrapping_add(origin_x,
                static_cast<std::int32_t>(random.next_u32() & 15U) + 8);
            const std::int32_t y = static_cast<std::int32_t>(
                random.next_u32() & 15U) + 55;
            const std::int32_t z = wrapping_add(origin_z,
                static_cast<std::int32_t>(random.next_u32() & 15U) + 8);
            place_outer_island(blocks, random, x, y, z);
            if ((random.next_u32() & 3U) == 0U) {
                place_outer_island(blocks, random, x, y, z);
            }
        }

        if (!outside_main_island(chunk_x, chunk_z)
            || island_height_value(chunk_x, chunk_z, 1, 1) <= 40.0F) {
            return;
        }

        const std::uint32_t plant_count = random.next_u32() % 5U;
        std::uint32_t x_word = random.next_u32();
        for (std::uint32_t index = 0; index < plant_count; ++index) {
            const std::uint32_t z_word = random.next_u32();
            const std::int32_t x = wrapping_add(origin_x,
                static_cast<std::int32_t>(x_word & 15U) + 8);
            const std::int32_t z = wrapping_add(origin_z,
                static_cast<std::int32_t>(z_word & 15U) + 8);
            const std::int32_t y = surface_y(blocks, x, z);
            if (y >= 0 && blocks.get(x, y, z) == 121U) {
                place_chorus(blocks, random, x, y + 1, z);
            }
            x_word = random.next_u32();
        }

        if (x_word % 700U == 0U) {
            const std::int32_t x = wrapping_add(origin_x,
                static_cast<std::int32_t>(random.next_u32() & 15U) + 8);
            const std::int32_t z = wrapping_add(origin_z,
                static_cast<std::int32_t>(random.next_u32() & 15U) + 8);
            const std::int32_t y = surface_y(blocks, x, z);
            if (y >= 0) {
                place_gateway(blocks, x,
                    y + static_cast<std::int32_t>(random.next_u32() % 7U) + 4,
                    z);
            }
        }
    }

    std::uint32_t world_seed_{};
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

std::array<std::uint8_t, Generator::block_count> Generator::generate_base_chunk(
    std::int32_t chunk_x, std::int32_t chunk_z) const {
    return impl_->generate(chunk_x, chunk_z);
}

std::array<std::uint8_t, Generator::block_count>
Generator::generate_fresh_entry_chunk(
    std::int32_t chunk_x,
    std::int32_t chunk_z) const {
    return impl_->generate_fresh_entry(chunk_x, chunk_z);
}

std::array<std::uint8_t, Generator::block_count>
Generator::generate_decorated_chunk(
    std::int32_t chunk_x,
    std::int32_t chunk_z) const {
    return impl_->generate_decorated(chunk_x, chunk_z);
}

std::array<std::uint8_t, Generator::block_count>
Generator::generate_fresh_visible_chunk(
    std::int32_t chunk_x,
    std::int32_t chunk_z) const {
    return impl_->generate_fresh_visible(chunk_x, chunk_z);
}

bool self_test() noexcept {
    struct TestCase final {
        std::uint32_t seed{};
        std::int32_t chunk_x{};
        std::int32_t chunk_z{};
        std::uint64_t expected_hash{};
    };
    constexpr std::array<TestCase, 4> cases{{
        {0U, 0, 0, 0xc99911f87ea3facaULL},
        {330675023U, 1, 0, 0xa2b8466c5b6e1bdfULL},
        {0x9733fc86U, -1, 1, 0xcf1599183f76be25ULL},
        {0xffff'ffffU, 6, -2, 0xa5a09890ac4621efULL},
    }};
    try {
        for (const TestCase& test : cases) {
            const Generator generator(test.seed);
            const auto blocks = generator.generate_base_chunk(
                test.chunk_x, test.chunk_z);
            std::uint64_t hash = 14'695'981'039'346'656'037ULL;
            for (const std::uint8_t block : blocks) {
                hash ^= block;
                hash *= 1'099'511'628'211ULL;
            }
            if (hash != test.expected_hash) {
                return false;
            }
        }

        // Static blocks observed after entering a new PE 1.1.5 End world.
        // These exercise the independently generated pillar scene rather
        // than relying on a hash of the raw density field alone.
        const Generator entry(330'675'023U);
        const auto spike = entry.generate_fresh_entry_chunk(2, 0);
        if (spike[block_index(10, 96, 0)] != kObsidian
            || spike[block_index(10, 97, 0)] != kBedrock
            || spike[block_index(10, 98, 0)] != kFire) {
            return false;
        }
        const auto platform = entry.generate_fresh_entry_chunk(6, 0);
        if (platform[block_index(4, 48, 0)] != kObsidian
            || platform[block_index(4, 49, 0)] != kAir) {
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

} // namespace pe115::end_terrain
