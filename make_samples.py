import os
import numpy as np
import soundfile as sf

os.makedirs("samples", exist_ok=True)
sr = 16000

# 1. Real Human Speech Simulation (Natural formants, varying pitch/energy)
t = np.linspace(0, 3.5, int(sr * 3.5))
# Dynamic human pitch contour (~140 Hz to 210 Hz inflection)
pitch = 140 + 40 * np.sin(2 * np.pi * 0.8 * t) + 15 * np.cos(2 * np.pi * 1.7 * t)
phase = 2 * np.pi * np.cumsum(pitch) / sr
human_wave = 0.4 * np.sin(phase) + 0.2 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
# Add natural speech pauses and micro-noise
envelope = np.clip(np.sin(2 * np.pi * 0.5 * t), 0.1, 1.0)
human_audio = (human_wave * envelope + np.random.normal(0, 0.005, len(t))).astype(np.float32)

sf.write("samples/real_human_speech.wav", human_audio, sr)
print("✅ Saved: samples/real_human_speech.wav")

# 2. Synthetic/AI    TTS Simulation (Unnaturally flat pitch & digital silence)
pitch_flat = 160.0  # Zero pitch variation
phase_flat = 2 * np.pi * pitch_flat * t
synth_wave = 0.5 * np.sin(phase_flat) + 0.25 * np.sin(2 * phase_flat)
synth_audio = synth_wave.astype(np.float32)

sf.write("samples/synth_ai_speech.wav", synth_audio, sr)
print("✅ Saved: samples/synth_ai_speech.wav")