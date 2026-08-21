# HearLink ASL — Real-time Speech-to-ASL Translation System

HearLink ASL is a production-grade AI system that translates spoken English or text input into American Sign Language (ASL) glosses and animates a 3D avatar with skeletal poses in real-time.

---

## 🌟 Key Features

1. **Transformer Translation Model:** Fine-tuned `t5-small` model trained on parallel English-to-ASL Gloss corpora (ASLG-PC12 and How2Sign).
2. **Sub-50ms Inference:** Exported model to **CTranslate2** with INT8 quantization for sub-50ms inference latency.
3. **Streaming ASR:** Integrated `Faster-Whisper` with Voice Activity Detection (VAD) over WebSockets.
4. **C++ WebAssembly Kinematics Engine:** `animation_engine.cpp` compiled with Emscripten (`-O3 -s WASM=1`) for high-performance SLERP rotation interpolation and CCD Inverse Kinematics.
5. **Interactive 3D Avatar:** Built with Three.js, rendering a 60 FPS rigged humanoid skeleton with facial blendshapes and dark-mode glassmorphism UI.

---

## 🏗️ Architecture

```
🎤 WebRTC Microphone Audio / Text Input
           │
           ▼
⚡ FastAPI WebSocket Stream Server (`app/main.py`)
           │
           ├──► 🎙️ Faster-Whisper Streaming ASR (`app/asr_engine.py`)
           ├──► 🧠 T5-small Gloss Translator (`app/gloss_translator.py`)
           └──► 🕺 Pose Keypoint Dispatcher (`app/pose_dispatcher.py`)
           │
           ▼
🌐 WebAssembly Kinematics Engine (`public/wasm/animation_engine.wasm`)
           │
           ▼
🎨 Three.js 3D Avatar Render Loop (`public/js/avatar.js`)
```

---

## 🚀 Quick Start Guide

### 1. Requirements & Setup

Ensure Python 3.10+ is installed on your system.

```bash
# Clone repository
git clone https://github.com/vigneshveereshwaran/mp3_to_ASL.git
cd mp3_to_ASL

# Install dependencies
python -m pip install -r requirements.txt
```

### 2. Dataset Preparation & Pose Library Generation

```bash
# Build sign library (26 fingerspelling letters + 57 common signs)
python pose_library/build_library.py

# Prepare parallel English -> Gloss training dataset
python datasets/download_and_prep.py
```

### 3. Model Fine-Tuning & Export (Optional)

```bash
# Fine-tune T5-small on English -> ASL Gloss pairs
python training/train_gloss_translator.py

# Evaluate model BLEU score
python training/evaluate_model.py

# Export model to CTranslate2 format
python training/export_model.py --benchmark
```

### 4. WASM Compilation (Optional)

If you have Emscripten SDK (`emsdk`) installed:

```bash
# Linux/macOS
bash src/build_wasm.sh

# Windows PowerShell
.\src\build_wasm.ps1
```

*(Note: Pre-bundled JS fallback interpolation is automatically active if WASM is not compiled.)*

### 5. Running the Application Server

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Now open your browser and navigate to:
👉 **`http://localhost:8000`**

---

## 🧪 Automated Testing

Run the full pytest suite:

```bash
python -m pytest tests/ -v
```

Tests include:
- `test_normalizer.py`: Text cleaning and ASL grammar transformations
- `test_translator.py`: Model translation fallback chain
- `test_pose_dispatcher.py`: Sign lookup and fingerspelling generator
- `test_e2e.py`: FastAPI REST API endpoints (`/health`, `/translate`, `/signs`)

---

## 📄 License

MIT License — free for research and personal use.
