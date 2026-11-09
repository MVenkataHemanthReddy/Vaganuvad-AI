"""
=============================================================================
Telugu <-> Hindi Speech Translation (v1 NEW — SeamlessM4T TTS)
=============================================================================
- SeamlessM4T does ASR + Translation + TTS (all in one, no Piper)
- W = Speak last translation (real Hindi/Telugu voice)
- S = Switch direction, P = Pause, Ctrl+C = Stop
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
        self.last_audio = None
        self.last_audio_sr = 16000
        self.last_text = ""
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
print("  1. Telugu → Hindi")
print("  2. Hindi → Telugu")
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
print(f"  ✅ {state.src_name} → {state.tgt_name}")

# =============================================================================
# LOAD
# =============================================================================

print(f"\n{'='*60}")
print("  LOADING...")
print("=" * 60)

t_start = time.perf_counter()

print("\n[1/2] Silero VAD...")
vad_model, _ = torch.hub.load("snakers4/silero-vad", model="silero_vad", force_reload=False, onnx=True)
print("  ✅ VAD ready")

print("\n[2/2] SeamlessM4T v2 large...")
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

print(f"\n  Loaded in {time.perf_counter()-t_start:.1f}s")

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
# AUDIO
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
    print(f"  🎙️  {state.src_name} → {state.tgt_name}")
    print(f"  S=Switch  P=Pause  W=Speak  Ctrl+C=Stop")
    print(f"{'='*60}\n")

    try:
        while state.running:
            # W key — play last audio
            if state.speak_requested:
                state.speak_requested = False
                if state.last_audio is not None:
                    print(f"  🔊 Speaking: {state.last_text[:60]}")
                    sd.play(state.last_audio, state.last_audio_sr)
                    sd.wait()
                else:
                    print("  🔇 Nothing yet")

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

                # === TRANSLATE (speech + text in one call) ===
                t0 = time.perf_counter()
                try:
                    aud = torch.from_numpy(utt).float().unsqueeze(0)
                    inp = processor(audio=aud, src_lang=src, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                    inp = {k: v.to(device) for k, v in inp.items()}

                    with torch.no_grad():
                        out = model.generate(
                            **inp,
                            tgt_lang=tgt,
                            generate_speech=True,
                            no_repeat_ngram_size=3,
                        )

                    # Text
                    if hasattr(out, 'sequences') and out.sequences is not None:
                        tids = out.sequences
                    else:
                        tids = out[2] if isinstance(out, tuple) and len(out) >= 3 else out[0]

                    text = processor.batch_decode(tids.cpu(), skip_special_tokens=True)[0].strip()

                    # Audio
                    audio_np = None
                    if hasattr(out, 'waveform') and out.waveform is not None:
                        audio_np = out.waveform.cpu().float().numpy().squeeze()
                    elif isinstance(out, tuple) and len(out) >= 1:
                        w = out[0]
                        if isinstance(w, torch.Tensor) and w.is_floating_point():
                            audio_np = w.cpu().float().numpy().squeeze()

                    sr = 16000
                    if hasattr(model.config, 'sampling_rate'):
                        sr = model.config.sampling_rate

                    if audio_np is not None:
                        with state.lock:
                            state.last_audio = audio_np
                            state.last_audio_sr = sr
                            state.last_text = text

                except Exception as e:
                    # Fallback: text only
                    try:
                        with torch.no_grad():
                            out = model.generate(**inp, tgt_lang=tgt, generate_speech=False, no_repeat_ngram_size=3)
                        tids = out.sequences if hasattr(out, 'sequences') else out[0]
                        text = processor.batch_decode(tids.cpu(), skip_special_tokens=True)[0].strip()
                        with state.lock:
                            state.last_text = text
                            state.last_audio = None
                    except Exception:
                        print(f"  ❌ {e}")
                        continue

                if not text:
                    continue

                ms = (time.perf_counter() - t0) * 1000
                has_audio = "W=play" if state.last_audio is not None else "no audio"
                print(f"  🔄 ({ms:.0f}ms): {text}  [{has_audio}]")

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
