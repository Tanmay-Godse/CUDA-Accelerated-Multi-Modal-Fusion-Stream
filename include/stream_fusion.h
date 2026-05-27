#pragma once

#include <cstddef>
#include <cstdint>

namespace csf {

inline constexpr int kAudioSampleRateHz = 16'000;
inline constexpr int kAudioChannels = 1;
inline constexpr int kDefaultAudioCapacityMs = 30'000;
inline constexpr int kDefaultAudioWindowMs = 3'000;

inline std::size_t audio_samples_for_duration_ms(int duration_ms, int sample_rate_hz) {
    if (duration_ms <= 0 || sample_rate_hz <= 0) {
        return 0;
    }

    const auto duration = static_cast<std::uint64_t>(duration_ms);
    const auto sample_rate = static_cast<std::uint64_t>(sample_rate_hz);
    return static_cast<std::size_t>((duration * sample_rate + 999ULL) / 1000ULL);
}

}  // namespace csf
