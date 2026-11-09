"""
=============================================================================
Telugu <-> Hindi + English Speech Translation (v8 — ALL WORKING)
=============================================================================
Shows target language + English translation
W = Speak English translation (Windows TTS)
S = Switch direction  P = Pause  Ctrl+C = Stop
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
import threading
import collections
import traceback
import numpy as np
import torch
import sounddevice as sd
import msvcrt
import pyttsx3

# =============================================================================
# CONFIG
# =============================================================================

SAMPLE_RATE     = 16000
BLOCK_SIZE      = 512
VAD_THRESHOLD   = 0.45
SILENCE_MS      = 600
MIN_SPEECH_MS   = 500
MAX_SPEECH_MS   = 15000

LANG_MAP = {
    "te": {"s2t": "tel", "name": "Telugu"},
    "hi": {"s2t": "hin", "name": "Hindi"},
}

# =============================================================================
# STATE
# =============================================================================

class State:
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt
        self.paused = False
        self.running = True
        self.speak_requested = False
        self.last_target_text = ""
        self.last_eng_text = ""
        self.lock = threading.Lock()

    def switch(self):
        with self.lock:
            self.src, self.tgt = self.tgt, self.src

    def toggle_pause(self):
        with self.lock:
            self.paused = not self.paused

    @property
    def src_s2t(self):
        return LANG_MAP[self.src]["s2t"]
    @property
    def tgt_s2t(self):
        return LANG_MAP[self.tgt]["s2t"]
    @property
    def src_name(self):
        return LANG_MAP[self.src]["name"]
    @property
    def tgt_name(self):
        return LANG_MAP[self.tgt]["name"]

# =============================================================================
# SELECT
# =============================================================================

print("=" * 60)
print("  1. Telugu → Hindi  (+English)")
print("  2. Hindi → Telugu  (+English)")
print("  (Switch anytime with 'S')")
print()

while True:
    c = input("  Enter 1 or 2: ").strip()
    if c == "1":
        state = State("te", "hi")
        break
    elif c == "2":
        state = State("hi", "te")
        break
print(f"  ✅ {state.src_name} → {state.tgt_name} + English")

# =============================================================================
# LOAD
# =============================================================================

print(f"\n{'='*60}")
print("  LOADING...")
print("=" * 60)

t_start = time.perf_counter()

# VAD
print("\n[1/3] Silero VAD...")
vad_model, _ = torch.hub.load("snakers4/silero-vad", model="silero_vad", force_reload=False, onnx=True)
print("  ✅ VAD ready")

# SeamlessM4T
print("\n[2/3] SeamlessM4T v2 large...")
from transformers import AutoProcessor, SeamlessM4Tv2Model
import warnings
warnings.filterwarnings("ignore")

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained(
    "facebook/seamless-m4t-v2-large", dtype=torch.float16,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
print(f"  ✅ Ready on {device}")

if torch.cuda.is_available():
    a = torch.cuda.memory_allocated() / 1024**3
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  VRAM: {a:.1f}/{t:.1f}GB")

# Windows TTS (for W key)
print("\n[3/3] Windows TTS (pyttsx3)...")
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 170)  # slightly slower for clarity
# Use best English voice available
voices = tts_engine.getProperty('voices')
for v in voices:
    if 'Zira' in v.name:  # Female US voice, clearer
        tts_engine.setProperty('voice', v.id)
        break
print(f"  ✅ Windows TTS ready ({len(voices)} voices)")

print(f"\n  All loaded in {time.perf_counter()-t_start:.1f}s")

# =============================================================================
# TRANSLATE HELPER
# =============================================================================

def translate(inputs_on_device, tgt_lang):
    """Text-only translation (fast, reliable)."""
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs_on_device,
                tgt_lang=tgt_lang,
                generate_speech=False,
                no_repeat_ngram_size=3,
            )
        if hasattr(out, 'sequences'):
            tids = out.sequences
        else:
            tids = out[0] if isinstance(out, tuple) else out
        return processor.batch_decode(tids.cpu(), skip_special_tokens=True)[0].strip()
    except Exception as e:
        return ""

# =============================================================================
# TTS (runs in separate thread to avoid blocking)
# =============================================================================

tts_lock = threading.Lock()

def speak_text(text):
    """Speak using Windows TTS."""
    if not text:
        print("  🔇 Nothing to speak yet")
        return
    try:
        with tts_lock:
            print(f"  🔊 Speaking: {text[:80]}")
            tts_engine.say(text)
            tts_engine.runAndWait()
    except Exception as e:
        print(f"  ❌ TTS error: {e}")

# =============================================================================
# KEYBOARD
# =============================================================================

def kb_listener():
    while state.running:
        try:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode("utf-8", errors="ignore").lower()
                if key == "s":
                    state.switch()
                    print(f"\n  🔄 SWITCHED → {state.src_name} → {state.tgt_name}\n")
                elif key == "p":
                    state.toggle_pause()
                    print(f"\n  {'⏸️ PAUSED' if state.paused else '▶️ RESUMED'}\n")
                elif key == "w":
                    with state.lock:
                        state.speak_requested = True
            time.sleep(0.05)
        except Exception:
            pass

# =============================================================================
# AUDIO CAPTURE
# =============================================================================

audio_buffer = collections.deque(maxlen=5000)
audio_lock = threading.Lock()

def audio_cb(indata, frames, ti, status):
    with audio_lock:
        audio_buffer.append(indata.copy())

def get_chunk():
    with audio_lock:
        return audio_buffer.popleft() if audio_buffer else None

# =============================================================================
# PIPELINE
# =============================================================================

def run():
    frames = []
    speaking = False
    sil_start = 0.0
    sp_start = 0.0
    fc = 0

    print(f"\n{'='*60}")
    print(f"  🎙️  {state.src_name} → {state.tgt_name} + English")
    print(f"  ─────────────────────────────────")
    print(f"  S = Switch    P = Pause/Resume")
    print(f"  W = Speak English out loud")
    print(f"  Ctrl+C = Stop")
    print(f"{'='*60}\n")

    try:
        while state.running:
            # W key — speak English translation
            if state.speak_requested:
                state.speak_requested = False
                # Speak in a thread so it doesn't block mic
                eng = state.last_eng_text
                if eng:
                    t = threading.Thread(target=speak_text, args=(eng,), daemon=True)
                    t.start()
                else:
                    print("  🔇 Say something first")

            if state.paused:
                get_chunk()
                time.sleep(0.01)
                continue

            chunk = get_chunk()
            if chunk is None:
                time.sleep(0.005)
                continue

            at = torch.from_numpy(chunk.flatten()).float()
            try:
                prob = vad_model(at, SAMPLE_RATE).item()
            except Exception:
                continue

            speech = prob > VAD_THRESHOLD
            now = time.perf_counter()
            fc += 1
            if fc % 1000 == 0:
                vad_model.reset_states()

            if speech:
                if not speaking:
                    speaking = True
                    sp_start = now
                    frames = []
                frames.append(chunk.copy())
                sil_start = 0.0
                if (now - sp_start) * 1000 <= MAX_SPEECH_MS:
                    continue

            if speaking:
                if speech and (now - sp_start) * 1000 <= MAX_SPEECH_MS:
                    continue
                if not speech:
                    frames.append(chunk.copy())
                    if sil_start == 0.0:
                        sil_start = now
                    if (now - sil_start) * 1000 < SILENCE_MS:
                        continue

                dur = (now - sp_start) * 1000
                speaking = False
                sil_start = 0.0
                vad_model.reset_states()

                if dur < MIN_SPEECH_MS:
                    frames = []
                    continue

                utt = np.concatenate(frames, axis=0).flatten()
                frames = []

                src = state.src_s2t
                tgt = state.tgt_s2t

                print(f"\n🎤 {dur:.0f}ms [{state.src_name}→{state.tgt_name}]")

                # Prepare input ONCE
                t0 = time.perf_counter()
                aud = torch.from_numpy(utt).float().unsqueeze(0)
                inp = processor(audio=aud, src_lang=src, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                inp_gpu = {k: v.to(device) for k, v in inp.items()}

                # 1. Target language translation
                target_text = translate(inp_gpu, tgt)
                ms1 = (time.perf_counter() - t0) * 1000

                if not target_text:
                    print("  ⏭️ Empty")
                    continue

                print(f"  🔄 {state.tgt_name} ({ms1:.0f}ms): {target_text}")

                # 2. English translation (re-prepare input for second call)
                t1 = time.perf_counter()
                inp2 = processor(audio=aud, src_lang=src, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                inp2_gpu = {k: v.to(device) for k, v in inp2.items()}
                eng_text = translate(inp2_gpu, "eng")
                ms2 = (time.perf_counter() - t1) * 1000

                if eng_text:
                    print(f"  🇬🇧 English ({ms2:.0f}ms): {eng_text}")

                total = ms1 + ms2
                print(f"  ⏱️  Total: {total:.0f}ms  |  W=speak English")

                # Save for W key
                with state.lock:
                    state.last_target_text = target_text
                    state.last_eng_text = eng_text if eng_text else target_text

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        state.running = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Done. ✅")

if __name__ == "__main__":
    threading.Thread(target=kb_listener, daemon=True).start()
    print("\nMic starting...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK_SIZE, callback=audio_cb):
        run()
