# SarcasTone 🎭
### Multimodal Sarcasm Detection Using Text and Speech Prosody

> *"Meaning is not only in what we say, but also in how we say it."*

---

## 📌 Project Overview
**SarcasTone** is an AI system that detects sarcasm in spoken dialogue by combining **lexical semantics** (the words spoken) with **acoustic speech prosody** (pitch variations, energy, tempo, and pauses).

A sentence like *"Oh wow, that was really amazing!"* is often misclassified as genuine praise by text-only models. SarcasTone listens to the vocal tone to catch deadpan delivery, flat pitch, or exaggerated mocking intonation to identify sarcastic intent.

---

## 🏗️ Architecture

```
[ Transcript / Speech Input ]       [ Audio Waveform (.wav) ]
            │                                  │
            ▼                                  ▼
    [ Text Preprocessing ]             [ Audio Preprocessing ]
  (DistilBERT Tokenizer)             (16kHz Resampling, Normalization)
            │                                  │
            ▼                                  ▼
    [ Text Encoder ]                  [ Prosodic Feature Extraction ]
   (DistilBERT 768-d)                (Pitch F0, Energy, Tempo, MFCCs)
            │                                  │
            └───────────────┬──────────────────┘
                            ▼
               [ Feature-Level Fusion Layer ]
               (Concatenate 768 + 26 = 794)
                            │
                            ▼
               [ LayerNorm + Dense + ReLU + Dropout ]
                            │
                            ▼
                  [ Binary Classifier ]
                 (Sarcastic vs Genuine)
```

---

## 📊 Experimental Results (Ablation Study)

Evaluation on test set comparing the 3 models as specified in the proposal:

| Metric | Text-Only Baseline | Audio-Only Baseline | Fusion Model (SarcasTone) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 60.0% | 60.0% | **80.0% ★** |
| **Precision**| 50.0% | 50.0% | **100.0% ★** |
| **Recall**   | 50.0% | 50.0% | **50.0% ★** |
| **F1-Score** | 50.0% | 50.0% | **66.7% ★** |

*★ Multimodal fusion achieves a +20% accuracy gain over unimodal baselines.*

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Retrain Models (Optional — Pretrained Weights Included)
```bash
python -m src.train
```

### 3. Run the Web App
```bash
python app.py
```
* The app automatically opens at `http://localhost:5000` in your browser.
* Click **"Click to Record"**, speak into your laptop microphone, and click **"Click to Stop"**.
* You can also test with the **"Quick Presets"** buttons without a microphone.

---

## 🔬 v2.0 Advanced Forensic Data Presentation

SarcasTone presents multimodal inference results through a clinical speech forensics dashboard:
1. **Circular Holographic Neon Gauge**: Dual-ring SVG probability and neural confidence meter with animated progress and status beacon.
2. **Diagnostic Sarcasm Archetype**: Classifies vocal delivery into distinct psychoacoustic personas (*The Deadpan Flatline*, *The Mocking Thespian*, *Passive-Aggressive Resignation*, *The Sardonic Smirk*, *Enthusiastic Warmth*, *Authentic Neutrality*).
3. **Interactive Lexical-Acoustic Saliency Matrix (Token Polarimeter)**: Token-by-token word inspector mapping lexical polarity against acoustic valence, quantifying semantic-acoustic contradiction with polarity inversion gaps ($\Delta$).
4. **6-Axis Sarcasm Radar Vector**: Interactive HTML5 Canvas spider plot mapping Pitch Flatness, Hyperbole, Vocal Drawl, Acoustic Contrast, Modal Divergence, and Irony Markers against a genuine human baseline.
5. **Synchronized Real-Time Laser Audio Playback**: Audio playback bar with a laser scrubber sweeping across the fundamental frequency ($F_0$) contour in real time, with click-to-seek functionality.
6. **13-Band MFCC Spectral Fingerprint**: Formant and resonance band visualization covering $F_1/F_2$ vowel formants, $F_3$ glottal tension, and high-frequency timbral air.
7. **Multimodal Latent Flow & Ablation Suite**: 794-dimensional tensor decomposition (DistilBERT 768-d + Librosa 26-d) with modal dominance indicators and unimodal vs multimodal performance ablation.
8. **Export & Report Suite**: One-click Markdown lab report copying and raw JSON forensic telemetry export.

---

## 📂 Project Structure

```
sarcas-tone/
├── app.py                      # Main entry point: starts Flask server & launches browser
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── SarcasTone_Proposal.pptx    # Original academic proposal
├── saved_models/               # Trained PyTorch model weights (.pt)
│   ├── text_baseline.pt
│   ├── audio_baseline.pt
│   └── fusion_model.pt
├── src/
│   ├── config.py               # Hyperparameters, paths & feature dimensions
│   ├── text_encoder.py         # DistilBERT contextual embedding
│   ├── audio_prosody.py        # Librosa pitch, energy, pace, MFCC & spectral extraction
│   ├── saliency_analyzer.py    # Lexical-acoustic token polarimeter & inversion index
│   ├── transcriber.py          # Whisper local speech-to-text
│   ├── emoji_analyzer.py       # Sarcastic emoji semantic lexicons
│   ├── data_loader.py          # Dataset splits & sample generator
│   ├── predictor.py            # End-to-end multimodal inference pipeline & archetypes
│   ├── train.py                # Model training & saving pipeline
│   ├── evaluate.py             # Accuracy, precision, recall, F1 & confusion matrices
│   ├── api.py                  # Flask REST API backend & audio stream endpoints
│   └── models/
│       ├── text_baseline.py    # Text-only baseline classifier
│       ├── audio_baseline.py   # Audio-only baseline classifier
│       └── fusion_model.py     # Multimodal feature-level fusion network (794-d)
├── data/
│   └── samples/                # Benchmark speech dataset & manifest
└── static/
    ├── index.html              # Enterprise multimodal speech analytics workbench
    ├── style.css               # Obsidian Pro design system, SVG meters & telemetry grids
    └── app.js                  # Synchronized audio scrubber, radar chart & token inspector
```

---

## 🔒 Enterprise Deployment & Licensing
* **Deployment**: Standalone Flask / WSGI service with local CPU/GPU PyTorch inference.
* **Security**: Zero external API dependencies during inference; all models execute strictly on-premise / localhost.
* **License**: Proprietary — All Rights Reserved. SarcasTone Systems Inc.
