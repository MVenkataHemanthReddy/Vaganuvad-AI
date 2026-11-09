"""
=============================================================================
Telugu <-> Hindi + English — Gradio UI with Noise Suppression
=============================================================================
NEW FEATURES:
  - DeepFilterNet/noisereduce: Filters background noise, only YOUR voice
  - Speaker Verification: Commented out, uncomment for panel demo
  - Mixed language: Auto-detects any language mix
  - Gradio UI: Buttons, history, clean look
  - Fully offline

Run: C:\\Users\\mitta\\hfenv\\Scripts\\python.exe pipeline_gradio.py
Opens: http://127.0.0.1:7860
=============================================================================
"""

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys
import time
import threading
import collections
import traceback
import subprocess
import numpy as np
import torch
import sounddevice as sd
import noisereduce as nr
import gradio as gr
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# SPEAKER VERIFICATION (COMMENTED OUT — UNCOMMENT FOR PANEL DEMO)
# =============================================================================
# HOW TO USE:
#   1. Uncomment the lines below (remove the # at the start)
#   2. Run the script — it will ask you to speak for 5 seconds
#   3. Your voice is saved as a profile
#   4. During panel: only YOUR voice gets translated, others are ignored
#
# from resemblyzer import VoiceEncoder, preprocess_wav
# speaker_encoder = VoiceEncoder()
# speaker_profile = None  # Will be set during enrollment
# SPEAKER_THRESHOLD = 0.75  # Similarity threshold (0.7-0.85)
#
# def enroll_speaker(audio_np, sr=16000):
#     """Record user's voice profile."""
#     global speaker_profile
#     wav = preprocess_wav(audio_np, source_sr=sr)
#     speaker_profile = speaker_encoder.embed_utterance(wav)
#     return True
#
# def is_target_speaker(audio_np, sr=16000):
#     """Check if audio matches enrolled speaker."""
#     if speaker_profile is None:
#         return True  # No profile = accept all
#     wav = preprocess_wav(audio_np, source_sr=sr)
#     embedding = speaker_encoder.embed_utterance(wav)
#     similarity = np.dot(speaker_profile, embedding)
#     return similarity > SPEAKER_THRESHOLD
#
# # === ENROLLMENT AT STARTUP (uncomment this block too) ===
# # print("🎤 Speak for 5 seconds to register your voice...")
# # enrollment_audio = []
# # def enroll_cb(indata, frames, ti, status):
# #     enrollment_audio.append(indata.copy())
# # with sd.InputStream(samplerate=16000, channels=1, dtype="float32",
# #                      blocksize=512, callback=enroll_cb):
# #     time.sleep(5)
# # full_audio = np.concatenate(enrollment_audio).flatten()
# # enroll_speaker(full_audio, 16000)
# # print("✅ Voice registered! Only your voice will be translated.")
# =============================================================================

# =============================================================================
# CONFIG
# =============================================================================

SAMPLE_RATE = 16000
BLOCK_SIZE = 512
VAD_THRESHOLD = 0.45
SILENCE_MS = 600
MIN_SPEECH_MS = 500
MAX_SPEECH_MS = 15000

INPUT_LANG_MAP = {
    "Auto-Detect (Any Spoken Language)": "auto",
    "Telugu (తెలుగు)": "tel",
    "Hindi (हिन्दी)": "hin",
    "English": "eng",
    "Tamil (தமிழ்)": "tam",
    "Kannada (ಕನ್ನಡ)": "kan",
    "Malayalam (മലയാളം)": "mal",
    "Bengali (বাংলা)": "ben",
    "Marathi (मराठी)": "mar",
    "Gujarati (ગુજરાતી)": "guj",
    "Punjabi (ਪੰਜਾਬੀ)": "pan",
    "Urdu (اردو)": "urd",
    "Odia (ଓଡ଼ିଆ)": "ory",
    "Spanish (Español)": "spa",
    "French (Français)": "fra",
    "German (Deutsch)": "deu",
    "Japanese (日本語)": "jpn",
    "Chinese (Mandarin)": "cmn",
    "Arabic (العربية)": "arb",
    "Russian (Русский)": "rus",
    "Korean (한국어)": "kor",
    "Italian (Italiano)": "ita",
    "Portuguese (Português)": "por",
}

TARGET_LANG_MAP = {
    "Hindi (हिन्दी)": "hin",
    "Telugu (తెలుగు)": "tel",
    "English": "eng",
    "Tamil (தமிழ்)": "tam",
    "Kannada (ಕನ್ನಡ)": "kan",
    "Malayalam (മലയാളം)": "mal",
    "Bengali (বাংলা)": "ben",
    "Marathi (मराठी)": "mar",
    "Gujarati (ગુજરાતી)": "guj",
    "Punjabi (ਪੰਜਾਬੀ)": "pan",
    "Urdu (اردو)": "urd",
    "Spanish (Español)": "spa",
    "French (Français)": "fra",
    "German (Deutsch)": "deu",
    "Japanese (日本語)": "jpn",
    "Chinese (Mandarin)": "cmn",
    "Arabic (العربية)": "arb",
    "Russian (Русский)": "rus",
    "Korean (한국어)": "kor",
}

LANG_MAP = {
    "te": {"s2t": "tel", "name": "Telugu"},
    "hi": {"s2t": "hin", "name": "Hindi"},
    "auto": {"s2t": "auto", "name": "Any Language"},
    "tel": {"s2t": "tel", "name": "Telugu"},
    "hin": {"s2t": "hin", "name": "Hindi"},
    "eng": {"s2t": "eng", "name": "English"},
}

# =============================================================================
# STATE
# =============================================================================

class State:
    def __init__(self):
        self.src = "auto"
        self.tgt = "hin"
        self.secondary_tgt = "eng"
        self.src_name = "Any Language (Auto)"
        self.tgt_name = "Hindi"
        self.secondary_name = "English"
        self.paused = False
        self.running = False
        self.last_target = ""
        self.last_secondary = ""
        self.history = []
        self.lock = threading.Lock()

    @property
    def last_english(self):
        return self.last_secondary

    @last_english.setter
    def last_english(self, val):
        self.last_secondary = val

    def switch_target(self):
        with self.lock:
            if self.tgt == "hin":
                self.tgt = "tel"
                self.tgt_name = "Telugu"
            else:
                self.tgt = "hin"
                self.tgt_name = "Hindi"

    def switch(self):
        # Backward-compatible toggle
        self.switch_target()

    @property
    def src_s2t(self):
        return self.src

    @property
    def tgt_s2t(self):
        return self.tgt

    @property
    def direction_str(self):
        return f"{self.src_name} → {self.tgt_name}"

state = State()

# =============================================================================
# LOAD MODELS
# =============================================================================

print("=" * 60)
print("  LOADING MODELS...")
print("=" * 60)

t_start = time.perf_counter()

# VAD
print("[1/2] Silero VAD...")
vad_model, _ = torch.hub.load("snakers4/silero-vad", model="silero_vad",
                                force_reload=False, onnx=True)
print("  ✅ VAD ready")

# SeamlessM4T
print("[2/2] SeamlessM4T v2 large...")
from transformers import AutoProcessor, SeamlessM4Tv2Model

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large",
                                           local_files_only=True)
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large",
                                            dtype=torch.float16,
                                            local_files_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
print(f"  ✅ Ready on {device}")

if torch.cuda.is_available():
    a = torch.cuda.memory_allocated() / 1024**3
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  VRAM: {a:.1f}/{t:.1f}GB")

print(f"\n  All loaded in {time.perf_counter()-t_start:.1f}s")
print("=" * 60)

# =============================================================================
# NOISE SUPPRESSION
# =============================================================================

def suppress_noise(audio_np, sr=16000):
    """Remove background noise using spectral gating."""
    try:
        cleaned = nr.reduce_noise(
            y=audio_np,
            sr=sr,
            stationary=True,
            prop_decrease=0.75,
            n_fft=512,
            hop_length=128,
        )
        return cleaned
    except Exception:
        return audio_np  # fallback: return original

# =============================================================================
# TRANSLATE
# =============================================================================

def translate(inputs_gpu, tgt_lang):
    """Text-only translation."""
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs_gpu,
                tgt_lang=tgt_lang,
                generate_speech=False,
                no_repeat_ngram_size=3,
            )
        if hasattr(out, 'sequences'):
            tids = out.sequences
        else:
            tids = out[0] if isinstance(out, tuple) else out
        return processor.batch_decode(tids.cpu(), skip_special_tokens=True)[0].strip()
    except Exception:
        return ""

# =============================================================================
# TTS (PowerShell — always works)
# =============================================================================

def speak_text(text):
    """Speak using Windows TTS via PowerShell."""
    if not text:
        return
    try:
        safe = text.replace("'", "''").replace('"', '`"')
        cmd = f"Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate=0; $s.Speak('{safe}')"
        subprocess.Popen(["powershell", "-Command", cmd],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

# =============================================================================
# AUDIO CAPTURE & PIPELINE
# =============================================================================

audio_buffer = collections.deque(maxlen=5000)
audio_lock = threading.Lock()

def audio_cb(indata, frames, ti, status):
    with audio_lock:
        audio_buffer.append(indata.copy())

def get_chunk():
    with audio_lock:
        return audio_buffer.popleft() if audio_buffer else None

mic_stream = None

def start_mic():
    global mic_stream
    if mic_stream is not None:
        return
    mic_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                 dtype="float32", blocksize=BLOCK_SIZE,
                                 callback=audio_cb)
    mic_stream.start()
    state.running = True

def stop_mic():
    global mic_stream
    state.running = False
    if mic_stream is not None:
        mic_stream.stop()
        mic_stream.close()
        mic_stream = None
    # Clear buffer
    with audio_lock:
        audio_buffer.clear()

# =============================================================================
# PIPELINE THREAD
# =============================================================================

pipeline_thread = None

def pipeline_loop(update_fn):
    """Main processing loop — runs in background thread."""
    frames = []
    speaking = False
    sil_start = 0.0
    sp_start = 0.0
    fc = 0

    while state.running:
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

            # === NOISE SUPPRESSION ===
            utt = suppress_noise(utt, SAMPLE_RATE)

            # === SPEAKER VERIFICATION (uncomment to enable) ===
            # if not is_target_speaker(utt, SAMPLE_RATE):
            #     continue  # Skip — not the registered speaker

            with state.lock:
                src = state.src
                tgt = state.tgt
                sec_tgt = state.secondary_tgt
                src_label = state.src_name
                tgt_label = state.tgt_name
                sec_label = state.secondary_name
                direction = state.direction_str

            # Audio tensor
            aud = torch.from_numpy(utt).float().unsqueeze(0)

            # Preprocess audio with processor:
            # When src is 'auto', omit src_lang so SeamlessM4T automatically identifies input speech language
            proc_kwargs = {"audio": aud, "sampling_rate": SAMPLE_RATE, "return_tensors": "pt"}
            if src and src != "auto":
                proc_kwargs["src_lang"] = src

            t0 = time.perf_counter()
            inp = processor(**proc_kwargs)
            inp_gpu = {k: v.to(device) for k, v in inp.items()}
            target_text = translate(inp_gpu, tgt)
            ms1 = (time.perf_counter() - t0) * 1000

            if not target_text:
                continue

            # Secondary translation (e.g. English or chosen secondary)
            sec_text = ""
            ms2 = 0
            if sec_tgt and sec_tgt != tgt:
                t1 = time.perf_counter()
                sec_text = translate(inp_gpu, sec_tgt)
                ms2 = (time.perf_counter() - t1) * 1000

            total_ms = ms1 + ms2

            print(f"DEBUG: [{src_label} -> {tgt_label}] target={target_text!r} | [{sec_label}]={sec_text!r}")

            # Save
            with state.lock:
                state.last_target = target_text
                state.last_secondary = sec_text or ""
                timestamp = time.strftime("%H:%M:%S")
                entry = (f"**{timestamp}** · {direction} · "
                         f"{dur:.0f}ms speech · {total_ms:.0f}ms translate\n\n"
                         f"🎯 **{tgt_label}:** {target_text}\n\n"
                         f"🌐 **{sec_label}:** {sec_text}")
                state.history.insert(0, entry)
                if len(state.history) > 50:
                    state.history.pop()

            # Trigger UI update
            if update_fn:
                try:
                    update_fn()
                except Exception:
                    pass

# =============================================================================
# GRADIO UI
# =============================================================================

def build_ui():
    update_event = threading.Event()

    def trigger_update():
        update_event.set()

    with gr.Blocks(
        title="Vaganuvad AI — Indic Speech Translator",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
        css="""
        .status-listening {
            background: linear-gradient(135deg, #22c55e, #16a34a) !important;
            color: white !important; font-size: 18px !important;
            padding: 12px !important; border-radius: 10px !important;
            text-align: center !important; font-weight: bold !important;
        }
        .status-stopped {
            background: linear-gradient(135deg, #ef4444, #dc2626) !important;
            color: white !important; font-size: 18px !important;
            padding: 12px !important; border-radius: 10px !important;
            text-align: center !important; font-weight: bold !important;
        }
        .status-paused {
            background: linear-gradient(135deg, #f59e0b, #d97706) !important;
            color: white !important; font-size: 18px !important;
            padding: 12px !important; border-radius: 10px !important;
            text-align: center !important; font-weight: bold !important;
        }
        .history-box {
            max-height: 400px; overflow-y: auto;
            font-family: 'Nirmala UI', 'Segoe UI', 'Arial Unicode MS', sans-serif !important;
            white-space: pre-wrap;
        }
        html, body, .gradio-container, .gradio-row, .gradio-column,
        .gr-text-input textarea, .gr-text-input input,
        .gradio-markdown, .history-box,
        .translation-box textarea, .translation-box input {
            font-family: 'Nirmala UI', 'Segoe UI', 'Arial Unicode MS', sans-serif !important;
        }
        .translation-box textarea, .translation-box input {
            font-size: 1rem !important;
            line-height: 1.5 !important;
        }
        """
    ) as app:
        gr.Markdown("# 🎙️ Vaganuvad AI (वागनुवाद) — Live Speech Translator")
        gr.Markdown("*Accepts Any Spoken Language · Real-Time Indic & Multilingual Translation · Fully Offline*")

        # Status bar
        status_box = gr.Markdown("🔴 **STOPPED**", elem_classes=["status-stopped"])

        # Direction display
        direction_display = gr.Markdown(
            f"### {state.direction_str} (+{state.secondary_name})",
            elem_id="direction"
        )

        # Language Selection Controls
        with gr.Row():
            src_dropdown = gr.Dropdown(
                choices=list(INPUT_LANG_MAP.keys()),
                value="Auto-Detect (Any Spoken Language)",
                label="🎤 Input Language",
                interactive=True,
            )
            tgt_dropdown = gr.Dropdown(
                choices=list(TARGET_LANG_MAP.keys()),
                value="Hindi (हिन्दी)",
                label="🎯 Primary Target Language",
                interactive=True,
            )
            sec_dropdown = gr.Dropdown(
                choices=list(TARGET_LANG_MAP.keys()),
                value="English",
                label="🌐 Secondary Target Language",
                interactive=True,
            )

        # Control buttons
        with gr.Row():
            start_btn = gr.Button("🎙️ Start Mic", variant="primary", size="lg")
            stop_btn = gr.Button("⏹️ Stop Mic", variant="stop", size="lg")
            switch_btn = gr.Button("🔄 Swap Telugu ⮂ Hindi", variant="secondary", size="lg")
            pause_btn = gr.Button("⏸️ Pause", variant="secondary", size="lg")

        # Translation output
        with gr.Row():
            with gr.Column():
                tgt_label_display = gr.Markdown(f"#### 🎯 {state.tgt_name} Translation")
                target_box = gr.Textbox(
                    label="", lines=3, interactive=False,
                    placeholder="Primary translation appears here...",
                    elem_classes=["translation-box"], elem_id="target_box"
                )
            with gr.Column():
                sec_label_display = gr.Markdown(f"#### 🌐 {state.secondary_name} Translation")
                secondary_box = gr.Textbox(
                    label="", lines=3, interactive=False,
                    placeholder="Secondary translation appears here...",
                    elem_classes=["translation-box"], elem_id="english_box"
                )

        # Speak buttons
        with gr.Row():
            speak_tgt_btn = gr.Button("🔊 Speak Primary Target", variant="primary", size="lg")
            speak_sec_btn = gr.Button("🔊 Speak Secondary Target", variant="primary", size="lg")

        # History
        gr.Markdown("---")
        gr.Markdown("### 📜 History")
        with gr.Row():
            history_box = gr.Markdown("*No translations yet...*", elem_classes=["history-box"], elem_id="history_box")
            clear_btn = gr.Button("🗑️ Clear", size="sm")

        # ---- Button handlers ----

        def on_change_src(new_src_name):
            with state.lock:
                state.src_name = "Auto (Any Language)" if "Auto" in new_src_name else new_src_name.split(" ")[0]
                state.src = INPUT_LANG_MAP.get(new_src_name, "auto")
            return f"### {state.direction_str} (+{state.secondary_name})"

        def on_change_tgt(new_tgt_name):
            with state.lock:
                state.tgt_name = new_tgt_name.split(" ")[0]
                state.tgt = TARGET_LANG_MAP.get(new_tgt_name, "hin")
            return (
                f"### {state.direction_str} (+{state.secondary_name})",
                f"#### 🎯 {state.tgt_name} Translation"
            )

        def on_change_sec(new_sec_name):
            with state.lock:
                state.secondary_name = new_sec_name.split(" ")[0]
                state.secondary_tgt = TARGET_LANG_MAP.get(new_sec_name, "eng")
            return (
                f"### {state.direction_str} (+{state.secondary_name})",
                f"#### 🌐 {state.secondary_name} Translation"
            )

        def on_start():
            global pipeline_thread
            start_mic()
            pipeline_thread = threading.Thread(target=pipeline_loop,
                                                args=(trigger_update,), daemon=True)
            pipeline_thread.start()
            return (
                gr.Markdown("🟢 **LISTENING**", elem_classes=["status-listening"]),
                f"### {state.direction_str} (+{state.secondary_name})"
            )

        def on_stop():
            stop_mic()
            return gr.Markdown("🔴 **STOPPED**", elem_classes=["status-stopped"])

        def on_switch():
            state.switch_target()
            new_val = "Telugu (తెలుగు)" if state.tgt == "tel" else "Hindi (हिन्दी)"
            status = "LISTENING" if state.running else "STOPPED"
            cls = "status-listening" if state.running else "status-stopped"
            if state.paused:
                status = "PAUSED"
                cls = "status-paused"
            return (
                new_val,
                f"### {state.direction_str} (+{state.secondary_name})",
                f"#### 🎯 {state.tgt_name} Translation",
                gr.Markdown(f"🔄 **{status} — Switched Target to {state.tgt_name}**",
                           elem_classes=[cls])
            )

        def on_pause():
            state.paused = not state.paused
            if state.paused:
                return gr.Markdown("⏸️ **PAUSED**", elem_classes=["status-paused"])
            else:
                return gr.Markdown("🟢 **LISTENING**", elem_classes=["status-listening"])

        def on_speak_sec():
            speak_text(state.last_secondary)

        def on_speak_tgt():
            speak_text(state.last_target)

        def on_clear():
            state.history.clear()
            return "*No translations yet...*"

        def get_updates():
            """Get latest translations for UI."""
            with state.lock:
                target = state.last_target
                sec = state.last_secondary
                hist = "\n\n---\n\n".join(state.history) if state.history else "*No translations yet...*"
            return target, sec, hist

        # Wire language selectors
        src_dropdown.change(on_change_src, inputs=[src_dropdown], outputs=[direction_display])
        tgt_dropdown.change(on_change_tgt, inputs=[tgt_dropdown], outputs=[direction_display, tgt_label_display])
        sec_dropdown.change(on_change_sec, inputs=[sec_dropdown], outputs=[direction_display, sec_label_display])

        # Wire buttons
        start_btn.click(on_start, outputs=[status_box, direction_display])
        stop_btn.click(on_stop, outputs=[status_box])
        switch_btn.click(on_switch, outputs=[tgt_dropdown, direction_display, tgt_label_display, status_box])
        pause_btn.click(on_pause, outputs=[status_box])
        speak_sec_btn.click(on_speak_sec)
        speak_tgt_btn.click(on_speak_tgt)
        clear_btn.click(on_clear, outputs=[history_box])

        # Auto-update translations using Timer
        timer = gr.Timer(1)
        timer.tick(get_updates, outputs=[target_box, secondary_box, history_box])

    return app

# =============================================================================
# START
# =============================================================================

if __name__ == "__main__":
    print("\n🌐 Starting Gradio UI...")
    print("   Open: http://127.0.0.1:7860\n")
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False,
               show_error=True)
