"""
Generate alarm sound for SentiGuard
"""
import numpy as np
from scipy.io import wavfile

# Parameters
SAMPLE_RATE = 44100
DURATION = 5.0  # 5 seconds

# Generate alarm sound (alternating frequencies)
t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))

# Create pulsing siren effect
freq1 = 800  # Hz
freq2 = 1000  # Hz
pulse_rate = 4  # pulses per second

# Alternating tone
signal = np.sin(2 * np.pi * freq1 * t) * (np.sin(2 * np.pi * pulse_rate * t) > 0).astype(float)
signal += np.sin(2 * np.pi * freq2 * t) * (np.sin(2 * np.pi * pulse_rate * t) <= 0).astype(float)

# Normalize and convert to 16-bit
signal = signal / np.max(np.abs(signal))
signal = (signal * 32767).astype(np.int16)

# Save
wavfile.write('static/alarm.wav', SAMPLE_RATE, signal)
print("✓ Alarm sound generated: static/alarm.wav")
