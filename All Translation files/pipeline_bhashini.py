"""
=============================================================================
Telugu <-> Hindi Speech Translation (BHASHINI — ONLINE VERSION)
=============================================================================
Uses Bhashini (Indian Govt) API — much better accuracy for Indian languages
Requires: Internet + API keys from https://bhashini.gov.in

HOW TO GET API KEYS (free):
  1. Go to https://bhashini.gov.in
  2. Register with email
  3. Go to Profile → API Keys
  4. Copy userId, ulcaApiKey, inferenceApiKey
  5. Paste below in API_CONFIG

Controls: S=Switch  P=Pause  W=Speak  Ctrl+C=Stop
=============================================================================
"""

import os
import sys
import time
import threading
import collections
import traceback
import base64
import struct
import numpy as np
import requests
import sounddevice as sd
import msvcrt
import subprocess
import torch

# =============================================================================
# ⚠️ PASTE YOUR BHASHINI API KEYS HERE ⚠️
# =============================================================================

API_CONFIG = {
    "userId": "YOUR_USER_ID_HERE",
    "ulcaApiKey": "YOUR_ULCA_API_KEY_HERE",
    "inferenceApiKey": "YOUR_INFERENCE_API_KEY_HERE",
}

# Pipeline search endpoint
PIPELINE_SEARCH_URL = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
PIPELINE_COMPUTE_URL = ""  # filled after pipeline search

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
    "te": {"name": "Telugu", "code": "te"},
    "hi": {"name": "Hindi", "code": "hi"},
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
        self.last_eng_text = ""
        self.last_target_text = ""
        self.lock = threading.Lock()
        # Bhashini pipeline config (filled after search)
        self.asr_service_id = ""
        self.nmt_service_id = ""
        self.tts_service_id = ""
        self.compute_url = ""

    def switch(self):
        with self.lock:
            self.src, self.tgt = self.tgt, self.src

    def toggle_pause(self):
        with self.lock:
            self.paused = not self.paused

    @property
    def src_name(self):
        return LANG_MAP[self.src]["name"]
    @property
    def tgt_name(self):
        return LANG_MAP[self.tgt]["name"]

# =============================================================================
# BHASHINI API
# =============================================================================

def search_pipeline(src_lang, tgt_lang):
    """Find Bhashini pipeline for ASR + NMT + TTS."""
    headers = {
        "Content-Type": "application/json",
        "ulcaApiKey": API_CONFIG["ulcaApiKey"],
        "userID": API_CONFIG["userId"],
    }

    payload = {
        "pipelineTasks": [
            {"taskType": "asr", "config": {"language": {"sourceLanguage": src_lang}}},
            {"taskType": "translation", "config": {"language": {"sourceLanguage": src_lang, "targetLanguage": tgt_lang}}},
            {"taskType": "tts", "config": {"language": {"sourceLanguage": tgt_lang}}},
        ],
        "pipelineRequestConfig": {"pipelineId": "64392f96daac500b55c543cd"},
    }

    resp = requests.post(PIPELINE_SEARCH_URL, json=payload, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    compute_url = data["pipelineInferenceAPIEndPoint"]["callbackUrl"]
    inference_key = data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]

    configs = data["pipelineResponseConfig"]
    asr_id = configs[0]["config"][0]["serviceId"]
    nmt_id = configs[1]["config"][0]["serviceId"]
    tts_id = configs[2]["config"][0]["serviceId"]

    return compute_url, inference_key, asr_id, nmt_id, tts_id


def bhashini_translate(audio_np, src_lang, tgt_lang, compute_url, inference_key, asr_id, nmt_id, tts_id):
    """Send audio to Bhashini, get translated text + TTS audio back."""

    # Convert float32 audio to 16-bit PCM WAV bytes, then base64
    audio_int16 = (audio_np * 32767).astype(np.int16)
    import io, wave
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": inference_key,
    }

    payload = {
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "language": {"sourceLanguage": src_lang},
                    "serviceId": asr_id,
                    "audioFormat": "wav",
                    "samplingRate": SAMPLE_RATE,
                },
            },
            {
                "taskType": "translation",
                "config": {
                    "language": {"sourceLanguage": src_lang, "targetLanguage": tgt_lang},
                    "serviceId": nmt_id,
                },
            },
            {
                "taskType": "tts",
                "config": {
                    "language": {"sourceLanguage": tgt_lang},
                    "serviceId": tts_id,
                    "gender": "female",
                    "samplingRate": SAMPLE_RATE,
                },
            },
        ],
        "inputData": {
            "input": [{"source": ""}],
            "audio": [{"audioContent": audio_b64}],
        },
    }

    resp = requests.post(compute_url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    result = resp.json()

    # Extract results
    asr_text = ""
    translated_text = ""
    tts_audio_b64 = ""

    outputs = result.get("pipelineResponse", [])
    for out in outputs:
        task = out.get("taskType", "")
        if task == "asr":
            asr_text = out["output"][0]["source"]
        elif task == "translation":
            translated_text = out["output"][0]["target"]
        elif task == "tts":
            tts_audio_b64 = out["audio"][0]["audioContent"]

    # Decode TTS audio
    tts_audio_np = None
    if tts_audio_b64:
        tts_bytes = base64.b64decode(tts_audio_b64)
        buf = io.BytesIO(tts_bytes)
        with wave.open(buf, 'rb') as wf:
            raw = wf.readframes(wf.getnframes())
            sr = wf.getframerate()
            tts_audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0

    return asr_text, translated_text, tts_audio_np

# =============================================================================
# SELECT
# =============================================================================

print("=" * 60)
print("  BHASHINI — ONLINE TRANSLATION (Higher Accuracy)")
print("=" * 60)
print()
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
# CHECK API KEYS
# =============================================================================

if "YOUR_" in API_CONFIG["userId"]:
    print("\n  ❌ API KEYS NOT SET!")
    print("  1. Register at https://bhashini.gov.in")
    print("  2. Get your userId, ulcaApiKey, inferenceApiKey")
    print("  3. Paste them in API_CONFIG at top of this file")
    sys.exit(1)

# =============================================================================
# LOAD
# =============================================================================

print(f"\n{'='*60}")
print("  LOADING...")
print("=" * 60)

t_start = time.perf_counter()

# VAD (local, no internet needed)
print("\n[1/2] Silero VAD...")
vad_model, _ = torch.hub.load("snakers4/silero-vad", model="silero_vad", force_reload=False, onnx=True)
print("  ✅ VAD ready")

# Bhashini pipeline search
print("\n[2/2] Connecting to Bhashini API...")
try:
    compute_url, inference_key, asr_id, nmt_id, tts_id = search_pipeline(state.src, state.tgt)
    state.compute_url = compute_url
    state.asr_service_id = asr_id
    state.nmt_service_id = nmt_id
    state.tts_service_id = tts_id
    # Store inference key for later
    API_CONFIG["current_inference_key"] = inference_key
    print(f"  ✅ Bhashini connected")
except Exception as e:
    print(f"  ❌ Bhashini connection failed: {e}")
    print("  Check your API keys and internet connection.")
    sys.exit(1)

print(f"\n  Ready in {time.perf_counter()-t_start:.1f}s")

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
                    # Re-search pipeline for new direction
                    try:
                        cu, ik, ai, ni, ti = search_pipeline(state.src, state.tgt)
                        state.compute_url = cu
                        state.asr_service_id = ai
                        state.nmt_service_id = ni
                        state.tts_service_id = ti
                        API_CONFIG["current_inference_key"] = ik
                    except Exception:
                        pass
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
    last_tts_audio = None

    print(f"\n{'='*60}")
    print(f"  🎙️  {state.src_name} → {state.tgt_name} (Bhashini)")
    print(f"  S=Switch  P=Pause  W=Speak  Ctrl+C=Stop")
    print(f"{'='*60}\n")

    try:
        while state.running:
            # W key — play last TTS audio
            if state.speak_requested:
                state.speak_requested = False
                if last_tts_audio is not None:
                    print(f"  🔊 Speaking: {state.last_target_text[:60]}")
                    sd.play(last_tts_audio, SAMPLE_RATE)
                    sd.wait()
                elif state.last_eng_text:
                    # Fallback: Windows TTS for English
                    safe = state.last_eng_text.replace("'", "''")
                    cmd = f"Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak('{safe}')"
                    subprocess.run(["powershell", "-Command", cmd], timeout=30, capture_output=True)
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

                print(f"\n🎤 {dur:.0f}ms [{state.src_name}→{state.tgt_name}]")

                # === BHASHINI API CALL ===
                t0 = time.perf_counter()
                try:
                    asr_text, translated, tts_audio = bhashini_translate(
                        utt, state.src, state.tgt,
                        state.compute_url,
                        API_CONFIG["current_inference_key"],
                        state.asr_service_id,
                        state.nmt_service_id,
                        state.tts_service_id,
                    )
                except Exception as e:
                    print(f"  ❌ API error: {e}")
                    continue

                ms = (time.perf_counter() - t0) * 1000

                if asr_text:
                    print(f"  📝 ASR: {asr_text}")
                if translated:
                    print(f"  🔄 {state.tgt_name}: {translated}")
                    has_audio = "W=play" if tts_audio is not None else ""
                    print(f"  ⏱️ {ms:.0f}ms  {has_audio}")

                # Store for W key
                last_tts_audio = tts_audio
                with state.lock:
                    state.last_target_text = translated or ""
                    state.last_eng_text = translated or asr_text or ""

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        state.running = False
        print("Done. ✅")

if __name__ == "__main__":
    threading.Thread(target=kb_listener, daemon=True).start()
    print("\nMic starting...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK_SIZE, callback=audio_cb):
        run()
