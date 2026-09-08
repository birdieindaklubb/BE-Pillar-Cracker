#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace pe115 {

constexpr std::size_t kPillarCount = 10U;
using PillarShapeMasks = std::array<std::uint16_t, kPillarCount>;
constexpr std::uint16_t kAllPillarShapes = (1U << kPillarCount) - 1U;

[[nodiscard]] constexpr PillarShapeMasks unconstrained_pillars() noexcept {
    PillarShapeMasks result{};
    result.fill(kAllPillarShapes);
    return result;
}

} // namespace pe115
