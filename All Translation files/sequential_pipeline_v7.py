"""
=============================================================================
Telugu <-> Hindi Speech Translation (v7 — SeamlessM4T v2)
=============================================================================
ONE MODEL DOES EVERYTHING: Speech → Translated Text (replaces Whisper + IndicTrans2)
Only Piper TTS remains for audio output.

First run downloads ~3.5GB model (cached after that).

Run:  C:\\Users\\mitta\\hfenv\\Scripts\\python.exe sequential_pipeline_v7.py
Stop: Ctrl+C
=============================================================================
"""

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import wave
import io
import threading
import collections
import traceback
import numpy as np
import torch
import sounddevice as sd
import torchaudio

# =============================================================================
# CONFIGURATION
# =============================================================================

SAMPLE_RATE     = 16000
BLOCK_SIZE      = 512

VAD_THRESHOLD   = 0.45
SILENCE_MS      = 600
MIN_SPEECH_MS   = 400
MAX_SPEECH_MS   = 15000

# SeamlessM4T language codes
S2T_LANG_MAP = {"te": "tel", "hi": "hin"}

# Piper TTS (still needed for speech output)
PIPER_VOICE_HI = r"C:\Users\mitta\Downloads\Desktop\en_US-lessac-medium.onnx"
PIPER_VOICE_TE = r"C:\Users\mitta\Downloads\Desktop\en_US-lessac-medium.onnx"

# =============================================================================
# STEP 0: SELECT LANGUAGE
# =============================================================================

print("=" * 60)
print("  SELECT TRANSLATION MODE")
print("=" * 60)
print()
print("  1. Telugu  → Hindi   (I will speak Telugu)")
print("  2. Hindi   → Telugu  (I will speak Hindi)")
print()

while True:
    choice = input("  Enter 1 or 2: ").strip()
    if choice == "1":
        SRC_LANG = "te"
        TGT_LANG = "hi"
        print("\n  ✅ Mode: Telugu → Hindi")
        break
    elif choice == "2":
        SRC_LANG = "hi"
        TGT_LANG = "te"
        print("\n  ✅ Mode: Hindi → Telugu")
        break
    else:
        print("  ❌ Please enter 1 or 2")

SRC_S2T = S2T_LANG_MAP[SRC_LANG]
TGT_S2T = S2T_LANG_MAP[TGT_LANG]

# =============================================================================
# STEP 1: LOAD MODELS
# =============================================================================

print(f"\n{'='*60}")
print("  LOADING MODELS...")
print("  First run downloads SeamlessM4T v2 (~3.5GB), cached after.")
print("=" * 60)

total_start = time.perf_counter()

# --- VAD ---
print("\n[1/3] Loading Silero VAD...")
t = time.perf_counter()
vad_model, _ = torch.hub.load("snakers4/silero-vad", model="silero_vad", force_reload=False, onnx=True)
print(f"  ✅ VAD ready ({time.perf_counter()-t:.1f}s)")

# --- SeamlessM4T v2 (replaces Whisper + IndicTrans2) ---
print("\n[2/3] Loading SeamlessM4T v2 (ASR + Translation in one)...")
t = time.perf_counter()
from transformers import AutoProcessor, SeamlessM4Tv2Model

s2t_processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
s2t_model = SeamlessM4Tv2Model.from_pretrained(
    "facebook/seamless-m4t-v2-large",
    dtype=torch.float16,  # FP16 to fit in 6GB VRAM
)

s2t_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
s2t_model = s2t_model.to(s2t_device)
s2t_model.eval()
print(f"  ✅ SeamlessM4T v2 ready on {s2t_device} ({time.perf_counter()-t:.1f}s)")

# --- Piper TTS ---
print("\n[3/3] Loading Piper TTS...")
t = time.perf_counter()
from piper import PiperVoice
piper_voice_hi = PiperVoice.load(PIPER_VOICE_HI)
piper_voice_te = PiperVoice.load(PIPER_VOICE_TE)
print(f"  ✅ Piper ready ({time.perf_counter()-t:.1f}s)")

# --- VRAM ---
if torch.cuda.is_available():
    alloc = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n  GPU: {torch.cuda.get_device_name(0)}, VRAM: {alloc:.1f}/{total:.1f}GB")

print(f"\n{'='*60}")
print(f"  MODELS LOADED in {time.perf_counter()-total_start:.1f}s")
print(f"  No more Whisper, no more IndicTrans2 errors!")
print(f"{'='*60}")

# =============================================================================
# AUDIO CAPTURE
# =============================================================================

audio_buffer = collections.deque(maxlen=5000)
audio_lock = threading.Lock()
is_running = True

def audio_callback(indata, frames, time_info, status):
    with audio_lock:
        audio_buffer.append(indata.copy())

def get_audio_chunk():
    with audio_lock:
        if audio_buffer:
            return audio_buffer.popleft()
    return None

# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline():
    global is_running

    speech_frames = []
    is_speaking = False
    silence_start = 0.0
    speech_start = 0.0
    frame_count = 0

    tts_sr = piper_voice_hi.config.sample_rate
    out_stream = sd.RawOutputStream(samplerate=tts_sr, channels=1, dtype="int16")
    out_stream.start()

    src_name = "Telugu" if SRC_LANG == "te" else "Hindi"
    tgt_name = "Hindi" if TGT_LANG == "hi" else "Telugu"

    print(f"\n{'='*60}")
    print(f"  🎙️  Speak {src_name} → {tgt_name}")
    print(f"  ASR+NMT: SeamlessM4T v2 (one model)")
    print(f"  Ctrl+C to stop")
    print(f"{'='*60}\n")

    try:
        while is_running:
            chunk = get_audio_chunk()
            if chunk is None:
                time.sleep(0.005)
                continue

            audio_tensor = torch.from_numpy(chunk.flatten()).float()
            try:
                speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()
            except Exception:
                continue

            is_speech = speech_prob > VAD_THRESHOLD
            now = time.perf_counter()

            frame_count += 1
            if frame_count % 1000 == 0:
                vad_model.reset_states()

            if is_speech:
                if not is_speaking:
                    is_speaking = True
                    speech_start = now
                    speech_frames = []
                speech_frames.append(chunk.copy())
                silence_start = 0.0
                if (now - speech_start) * 1000 > MAX_SPEECH_MS:
                    pass
                else:
                    continue

            if is_speaking:
                if is_speech and (now - speech_start) * 1000 <= MAX_SPEECH_MS:
                    continue
                if not is_speech:
                    speech_frames.append(chunk.copy())
                    if silence_start == 0.0:
                        silence_start = now
                    if (now - silence_start) * 1000 < SILENCE_MS:
                        continue

                dur_ms = (now - speech_start) * 1000
                is_speaking = False
                silence_start = 0.0
                vad_model.reset_states()

                if dur_ms < MIN_SPEECH_MS:
                    speech_frames = []
                    continue

                utterance = np.concatenate(speech_frames, axis=0).flatten()
                speech_frames = []
                print(f"\n🎤 Speech: {dur_ms:.0f}ms")

                # ==============================================
                # SeamlessM4T: Speech → Translated Text (ONE STEP)
                # No separate ASR, no separate NMT. Just one call.
                # ==============================================
                t0 = time.perf_counter()
                try:
                    # Prepare audio for SeamlessM4T
                    audio_for_model = torch.from_numpy(utterance).float().unsqueeze(0)

                    inputs = s2t_processor(
                        audio=audio_for_model,
                        src_lang=SRC_S2T,
                        sampling_rate=SAMPLE_RATE,
                        return_tensors="pt",
                    )
                    # Move to GPU
                    inputs = {k: v.to(s2t_device) for k, v in inputs.items()}

                    with torch.no_grad():
                        output = s2t_model.generate(
                            **inputs,
                            tgt_lang=TGT_S2T,
                            generate_speech=False,
                        )

                    # output is GenerateEncoderDecoderOutput — get .sequences
                    if hasattr(output, 'sequences'):
                        token_ids = output.sequences
                    elif isinstance(output, tuple) and hasattr(output[0], 'sequences'):
                        token_ids = output[0].sequences
                    elif isinstance(output, tuple):
                        token_ids = output[0]
                    else:
                        token_ids = output

                    translated = s2t_processor.batch_decode(
                        token_ids.cpu(), skip_special_tokens=True
                    )[0].strip()

                except Exception as e:
                    print(f"  ❌ Translation error: {e}")
                    traceback.print_exc()
                    continue

                if not translated:
                    print("  ⏭️ Empty translation — skipping")
                    continue

                translate_ms = (time.perf_counter() - t0) * 1000
                print(f"  🔄 Translated ({translate_ms:.0f}ms): {translated}")

                # === TTS (Piper) ===
                t0 = time.perf_counter()
                voice = piper_voice_hi if TGT_LANG == "hi" else piper_voice_te
                try:
                    buf = io.BytesIO()
                    with wave.open(buf, "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(voice.config.sample_rate)
                        voice.synthesize(translated, w)
                    buf.seek(0)
                    with wave.open(buf, "rb") as r:
                        raw_bytes = r.readframes(r.getnframes())
                    if raw_bytes:
                        out_stream.write(raw_bytes)
                except Exception as e:
                    print(f"  ❌ TTS error: {e}")
                    continue

                tts_ms = (time.perf_counter() - t0) * 1000
                total_ms = translate_ms + tts_ms
                print(f"  🔊 TTS ({tts_ms:.0f}ms) | Total: {total_ms:.0f}ms")

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        is_running = False
        out_stream.stop()
        out_stream.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Done. ✅")

if __name__ == "__main__":
    print("\nStarting microphone...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK_SIZE, callback=audio_callback):
        run_pipeline()
