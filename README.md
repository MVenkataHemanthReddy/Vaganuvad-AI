# 🎙️ Vaganuvad AI (वागनुवाद) — Real-Time Indic Speech Translator

**Vaganuvad AI** is an offline, low-latency, real-time speech-to-text and speech-to-speech translation pipeline bridging **Telugu** and **Hindi**, with simultaneous **English** output. Built with Meta's **SeamlessM4T v2 Large**, **Silero VAD**, and an interactive **Gradio** web interface.

---

## 🌟 Key Features

- **Bidirectional Live Translation**: Instant bidirectional translation between **Telugu (తెలుగు)** and **Hindi (हिन्दी)**.
- **Dual Output**: Provides translation in the target Indic language alongside an immediate English translation.
- **Noise Suppression**: Built-in spectral gating noise reduction via `noisereduce` to eliminate background hums and environmental audio noise.
- **Voice Activity Detection (VAD)**: Powered by ONNX-optimized **Silero VAD** for ultra-fast silence detection and utterance chunking.
- **Fully Offline**: Runs completely on local models without requiring cloud APIs or internet connectivity once weights are cached.
- **Interactive Gradio UI**: Complete control dashboard with Start/Stop Mic, Direction Switch, Pause, Speech History, and Text-to-Speech playback.
- **Speaker Verification Support**: Ready-to-enable speaker verification via `resemblyzer` for panel demos.

---

## 🏗️ Architecture

```
[ Microphone (16kHz) ]
          │
          ▼
   [ Silero VAD ]  ───(Silence / Noise)───► Discard
          │
      (Speech)
          ▼
[ Noise Suppression (Spectral Gating) ]
          │
          ▼
 [ SeamlessM4T v2 Large (Audio Encoder) ]
          │
          ├───► [ Target Language Generation (Telugu / Hindi) ]
          └───► [ English Translation Generation ]
          │
          ▼
[ Gradio UI + Windows TTS Playback ]
```

---

## 📋 Requirements & Prerequisites

- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.10+
- **Hardware**: GPU recommended (NVIDIA with CUDA and >= 6GB VRAM) for real-time inference with float16. Can also run on CPU.
- **Audio**: Working microphone and speakers.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/MVenkataHemanthReddy/Vaganuvad-AI.git
cd Vaganuvad-AI
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the Gradio App
```bash
python pipeline_gradio.py
```
Open your browser at `http://127.0.0.1:7860`.

---

## 🎮 Interface Controls

| Control | Action |
|---|---|
| **🎙️ Start Mic** | Activates the microphone and begins real-time speech translation loop |
| **⏹️ Stop Mic** | Stops microphone stream and pauses the inference thread |
| **🔄 Switch** | Switches translation direction (`Telugu → Hindi` ⮂ `Hindi → Telugu`) |
| **⏸️ Pause** | Temporarily pauses processing without clearing the state |
| **🔊 Speak Target / English** | Speaks aloud the translation via local Windows speech synthesizer |
| **🗑️ Clear** | Clears the session translation history |

---

## 📁 Repository Structure

```
├── pipeline_gradio.py          # Primary Gradio UI pipeline with noise reduction
├── sequential_pipeline_v8.py   # CLI version with keyboard triggers
├── sequential_pipeline_v7.py   # Previous iteration
├── requirements.txt            # Python dependencies
├── push_to_github.bat          # Automated GitHub push helper
└── All Translation files/       # Version iterations and experiments
```

---

## 📜 License
MIT License. Free for academic, research, and open-source use.
