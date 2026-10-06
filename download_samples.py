import os
import soundfile as sf
from datasets import load_dataset

os.makedirs("samples", exist_ok=True)

print("Fetching real Malayalam audio sample...")
try:
    ds = load_dataset("mozilla-foundation/common_voice_13_0", "ml", split="train", streaming=True)
    sample = next(iter(ds))
    sf.write("samples/real_malayalam.wav", sample["audio"]["array"], sample["audio"]["sampling_rate"])
    print("✅ Successfully saved to samples/real_malayalam.wav")
except Exception as e:
    print(f"❌ Failed to download sample: {e}")