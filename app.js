/**
 * SarcasTone™ Enterprise — Multimodal Speech Sarcasm Intelligence
 * Production Client Controller & Forensic Telemetry Engine
 * WCAG 2.1 AA Compliant · Zero-Egress Local Architecture
 */

// ===== DOM Elements =====
const recordBtn = document.getElementById("recordBtn");
const fileUpload = document.getElementById("fileUpload");
const recordTimer = document.getElementById("recordTimer");
const recordingBadge = document.getElementById("recordingBadge");
const recStatusText = document.getElementById("recStatusText");
const waveformCanvas = document.getElementById("waveformCanvas");
const visualizerOverlay = document.getElementById("visualizerOverlay");
const loadingSection = document.getElementById("loadingSection");
const resultsSection = document.getElementById("resultsSection");
const resetBtn = document.getElementById("resetBtn");
const systemStatus = document.getElementById("systemStatus");

// VAD Elements
const vadLevel = document.getElementById("vadLevel");
const vadStatus = document.getElementById("vadStatus");

// Primary Verdict Elements
const verdictHeroCard = document.getElementById("verdictHeroCard");
const ringProgress = document.getElementById("ringProgress");
const sarcasmProb = document.getElementById("sarcasmProb");
const confidenceValue = document.getElementById("confidenceValue");
const verdictBeacon = document.getElementById("verdictBeacon");
const archetypeIcon = document.getElementById("archetypeIcon");
const archetypeName = document.getElementById("archetypeName");
const archetypeDescription = document.getElementById("archetypeDescription");
const conflictChip = document.getElementById("conflictChip");
const verdictHeadline = document.getElementById("verdictHeadline");

// Audio Playback Elements
const coreAudioElement = document.getElementById("coreAudioElement");
const audioPlayToggle = document.getElementById("audioPlayToggle");
const heroPlayBtn = document.getElementById("heroPlayBtn");
const playIcon = document.getElementById("playIcon");
const currentTimeEl = document.getElementById("currentTime");
const totalDurationEl = document.getElementById("totalDuration");
const audioScrubber = document.getElementById("audioScrubber");
const pitchLaserCursor = document.getElementById("pitchLaserCursor");

// Actions & Toast
const copyReportBtn = document.getElementById("copyReportBtn");
const exportJsonBtn = document.getElementById("exportJsonBtn");
const toastNotification = document.getElementById("toastNotification");
const toastMessage = document.getElementById("toastMessage");

// Saliency Elements
const tokenStream = document.getElementById("tokenStream");
const tokenInspectorBox = document.getElementById("tokenInspectorBox");
const lexicalValence = document.getElementById("lexicalValence");
const lexicalTag = document.getElementById("lexicalTag");
const acousticValence = document.getElementById("acousticValence");
const acousticTag = document.getElementById("acousticTag");
const polarityGap = document.getElementById("polarityGap");
const polarityGapTag = document.getElementById("polarityGapTag");
const saliencyNarrative = document.getElementById("saliencyNarrative");
const emojiRecognitionBar = document.getElementById("emojiRecognitionBar");
const emojiTags = document.getElementById("emojiTags");

// Radar Elements
const radarCanvas = document.getElementById("radarCanvas");
const radarCtx = radarCanvas.getContext("2d");
const valFlatness = document.getElementById("valFlatness");
const valHyperbole = document.getElementById("valHyperbole");
const valDrawl = document.getElementById("valDrawl");
const valContrast = document.getElementById("valContrast");
const valDivergence = document.getElementById("valDivergence");
const valMarkers = document.getElementById("valMarkers");

// Pitch Track Elements
const pitchContourCanvas = document.getElementById("pitchContourCanvas");
const pitchCanvasCtx = pitchContourCanvas.getContext("2d");

// Telemetry Elements
const pitchBadge = document.getElementById("pitchBadge");
const pitchValue = document.getElementById("pitchValue");
const pitchStdVal = document.getElementById("pitchStdVal");
const pitchRangeVal = document.getElementById("pitchRangeVal");
const pitchDetail = document.getElementById("pitchDetail");

const energyBadge = document.getElementById("energyBadge");
const energyValue = document.getElementById("energyValue");
const energyPeakVal = document.getElementById("energyPeakVal");
const energyDetail = document.getElementById("energyDetail");

const paceBadge = document.getElementById("paceBadge");
const tempoValue = document.getElementById("tempoValue");
const drawlVal = document.getElementById("drawlVal");
const paceDetail = document.getElementById("paceDetail");

const timbreBadge = document.getElementById("timbreBadge");
const centroidValue = document.getElementById("centroidValue");
const zcrVal = document.getElementById("zcrVal");
const timbreDetail = document.getElementById("timbreDetail");

// MFCC & Modality
const mfccBarsContainer = document.getElementById("mfccBarsContainer");
const dominanceChip = document.getElementById("dominanceChip");
const textInfluenceScore = document.getElementById("textInfluenceScore");
const textInfluenceBar = document.getElementById("textInfluenceBar");
const audioInfluenceScore = document.getElementById("audioInfluenceScore");
const audioInfluenceBar = document.getElementById("audioInfluenceBar");
const conflictScore = document.getElementById("conflictScore");
const conflictMeter = document.getElementById("conflictMeter");
const conflictDetail = document.getElementById("conflictDetail");

// Ablation Elements
const textMeter = document.getElementById("textMeter");
const textScore = document.getElementById("textScore");
const textChip = document.getElementById("textChip");
const audioMeter = document.getElementById("audioMeter");
const audioScore = document.getElementById("audioScore");
const audioChip = document.getElementById("audioChip");
const fusionMeter = document.getElementById("fusionMeter");
const fusionScore = document.getElementById("fusionScore");
const fusionChip = document.getElementById("fusionChip");

// Sandbox Elements
const transcriptInput = document.getElementById("transcriptInput");
const reevaluateBtn = document.getElementById("reevaluateBtn");

// ===== State Management =====
let isRecording = false;
let timerInterval = null;
let recordingStartTime = 0;
let audioContext = null;
let analyser = null;
let scriptProcessor = null;
let mediaStream = null;
let pcmBuffers = [];
let animationId = null;

// VAD Parameters
const SPEECH_RMS_THRESHOLD = 0.024;
const SILENCE_RMS_THRESHOLD = 0.012;
const AUTO_STOP_SILENCE_MS = 1600;
let hasSpoken = false;
let silenceStartTime = null;

// Telemetry Caches
let currentAnalysisData = null;
let currentPitchContour = [];
let isPitchCurrentlyFlat = false;
let toastTimeout = null;

// ===== Waveform Visualizer & Canvas Setup =====
const canvasCtx = waveformCanvas.getContext("2d");

function setupCanvases() {
    const dpr = window.devicePixelRatio || 1;

    // Waveform Canvas
    const wRect = waveformCanvas.getBoundingClientRect();
    if (wRect.width > 0) {
        waveformCanvas.width = Math.floor(wRect.width * dpr);
        waveformCanvas.height = Math.floor(wRect.height * dpr);
        canvasCtx.setTransform(1, 0, 0, 1, 0, 0);
        canvasCtx.scale(dpr, dpr);
        drawIdleWaveform();
    }

    // Pitch Contour Canvas
    const pRect = pitchContourCanvas.getBoundingClientRect();
    if (pRect.width > 0) {
        pitchContourCanvas.width = Math.floor(pRect.width * dpr);
        pitchContourCanvas.height = Math.floor(pRect.height * dpr);
        pitchCanvasCtx.setTransform(1, 0, 0, 1, 0, 0);
        pitchCanvasCtx.scale(dpr, dpr);
        if (currentPitchContour.length > 0) {
            drawPitchContour(currentPitchContour, isPitchCurrentlyFlat);
        } else {
            drawIdlePitchGrid();
        }
    }

    // Radar Chart Canvas
    const rRect = radarCanvas.getBoundingClientRect();
    if (rRect.width > 0) {
        radarCanvas.width = Math.floor(rRect.width * dpr);
        radarCanvas.height = Math.floor(rRect.height * dpr);
        radarCtx.setTransform(1, 0, 0, 1, 0, 0);
        radarCtx.scale(dpr, dpr);
        if (currentAnalysisData && currentAnalysisData.radar_data) {
            drawRadarChart(currentAnalysisData.radar_data);
        } else {
            drawIdleRadar();
        }
    }
}

function drawIdleWaveform() {
    const w = waveformCanvas.getBoundingClientRect().width;
    const h = waveformCanvas.getBoundingClientRect().height;
    if (w === 0 || h === 0) return;

    canvasCtx.clearRect(0, 0, w, h);

    // Subtle grid lines
    canvasCtx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    canvasCtx.lineWidth = 1;
    for (let y = 0; y <= h; y += 24) {
        canvasCtx.beginPath();
        canvasCtx.moveTo(0, y);
        canvasCtx.lineTo(w, y);
        canvasCtx.stroke();
    }

    // Center idle line
    canvasCtx.strokeStyle = "rgba(59, 130, 246, 0.35)";
    canvasCtx.lineWidth = 1.5;
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, h / 2);
    canvasCtx.lineTo(w, h / 2);
    canvasCtx.stroke();
}

function drawLiveWaveform() {
    if (!analyser) return;

    const bufferLength = analyser.fftSize;
    const dataArray = new Float32Array(bufferLength);
    analyser.getFloatTimeDomainData(dataArray);

    const w = waveformCanvas.getBoundingClientRect().width;
    const h = waveformCanvas.getBoundingClientRect().height;

    canvasCtx.clearRect(0, 0, w, h);

    // Grid
    canvasCtx.strokeStyle = "rgba(255, 255, 255, 0.03)";
    canvasCtx.lineWidth = 1;
    for (let y = 0; y <= h; y += 24) {
        canvasCtx.beginPath();
        canvasCtx.moveTo(0, y);
        canvasCtx.lineTo(w, y);
        canvasCtx.stroke();
    }

    // Live Oscilloscope Stroke
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = "#3b82f6";
    canvasCtx.beginPath();

    const sliceWidth = w / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i];
        const y = (v * (h * 0.44)) + (h / 2);
        if (i === 0) canvasCtx.moveTo(x, y);
        else canvasCtx.lineTo(x, y);
        x += sliceWidth;
    }
    canvasCtx.stroke();

    // Compute RMS Energy for VAD
    let sumSquares = 0;
    for (let i = 0; i < bufferLength; i++) {
        sumSquares += dataArray[i] * dataArray[i];
    }
    const currentRMS = Math.sqrt(sumSquares / bufferLength);
    const levelPct = Math.min(100, Math.round(currentRMS * 450));
    vadLevel.style.width = `${levelPct}%`;

    const now = Date.now();

    if (currentRMS >= SPEECH_RMS_THRESHOLD) {
        hasSpoken = true;
        silenceStartTime = null;
        vadStatus.textContent = "Voice detected · Streaming 16kHz PCM";
        vadStatus.style.color = "#6ee7b7";
        recStatusText.textContent = "VOICE ACTIVE";
    } else if (hasSpoken && currentRMS < SILENCE_RMS_THRESHOLD) {
        if (!silenceStartTime) {
            silenceStartTime = now;
        }

        const elapsedSilence = now - silenceStartTime;
        const remaining = Math.max(0, ((AUTO_STOP_SILENCE_MS - elapsedSilence) / 1000)).toFixed(1);

        if (elapsedSilence >= AUTO_STOP_SILENCE_MS) {
            vadStatus.textContent = "Voice boundary reached · Ingestion finalizing";
            vadStatus.style.color = "#94a3b8";
            stopRecording();
            return;
        } else {
            vadStatus.textContent = `Pause detected · completing in ${remaining}s`;
            vadStatus.style.color = "#f59e0b";
            recStatusText.textContent = "PAUSE DETECTED";
        }
    } else {
        if (!hasSpoken) {
            vadStatus.textContent = "Awaiting speech onset...";
            vadStatus.style.color = "#94a3b8";
            recStatusText.textContent = "STREAMING AUDIO";
        }
    }

    animationId = requestAnimationFrame(drawLiveWaveform);
}

// ===== Prosody Radar Chart Engine =====
const RADAR_AXES = [
    { key: "pitch_flatness", label: "Pitch Flatness" },
    { key: "hyperbolic_intensity", label: "Hyperbole" },
    { key: "vocal_drawl", label: "Vocal Drawl" },
    { key: "acoustic_contrast", label: "Contrast" },
    { key: "cross_modal_divergence", label: "Divergence" },
    { key: "ironic_marker_load", label: "Irony Markers" }
];

function drawIdleRadar() {
    const w = radarCanvas.getBoundingClientRect().width;
    const h = radarCanvas.getBoundingClientRect().height;
    if (w === 0 || h === 0) return;

    radarCtx.clearRect(0, 0, w, h);
    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(cx, cy) - 28;

    drawRadarRings(cx, cy, radius);
}

function drawRadarRings(cx, cy, radius) {
    const numAxes = RADAR_AXES.length;
    const rings = [0.33, 0.66, 1.0];

    for (const rFraction of rings) {
        radarCtx.beginPath();
        radarCtx.strokeStyle = "rgba(255, 255, 255, 0.08)";
        radarCtx.lineWidth = 1;

        for (let i = 0; i < numAxes; i++) {
            const angle = (Math.PI * 2 / numAxes) * i - Math.PI / 2;
            const x = cx + Math.cos(angle) * (radius * rFraction);
            const y = cy + Math.sin(angle) * (radius * rFraction);
            if (i === 0) radarCtx.moveTo(x, y);
            else radarCtx.lineTo(x, y);
        }
        radarCtx.closePath();
        radarCtx.stroke();
    }

    // Axes Lines & Labels
    radarCtx.font = "500 10px 'JetBrains Mono', monospace";
    radarCtx.fillStyle = "#64748b";
    radarCtx.textAlign = "center";
    radarCtx.textBaseline = "middle";

    for (let i = 0; i < numAxes; i++) {
        const angle = (Math.PI * 2 / numAxes) * i - Math.PI / 2;
        const x = cx + Math.cos(angle) * radius;
        const y = cy + Math.sin(angle) * radius;

        radarCtx.beginPath();
        radarCtx.strokeStyle = "rgba(255, 255, 255, 0.1)";
        radarCtx.moveTo(cx, cy);
        radarCtx.lineTo(x, y);
        radarCtx.stroke();

        const labelX = cx + Math.cos(angle) * (radius + 18);
        const labelY = cy + Math.sin(angle) * (radius + 18);
        radarCtx.fillText(RADAR_AXES[i].label, labelX, labelY);
    }
}

function drawRadarChart(radarData) {
    const w = radarCanvas.getBoundingClientRect().width;
    const h = radarCanvas.getBoundingClientRect().height;
    if (w === 0 || h === 0) return;

    radarCtx.clearRect(0, 0, w, h);
    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(cx, cy) - 30;
    const numAxes = RADAR_AXES.length;

    drawRadarRings(cx, cy, radius);

    // Baseline Polygon (Normal genuine speech reference)
    const baselineScores = [25, 20, 30, 25, 15, 15];
    radarCtx.beginPath();
    radarCtx.setLineDash([4, 4]);
    radarCtx.strokeStyle = "rgba(16, 185, 129, 0.4)";
    radarCtx.lineWidth = 1.5;

    for (let i = 0; i < numAxes; i++) {
        const val = baselineScores[i] / 100;
        const angle = (Math.PI * 2 / numAxes) * i - Math.PI / 2;
        const x = cx + Math.cos(angle) * (radius * val);
        const y = cy + Math.sin(angle) * (radius * val);
        if (i === 0) radarCtx.moveTo(x, y);
        else radarCtx.lineTo(x, y);
    }
    radarCtx.closePath();
    radarCtx.stroke();
    radarCtx.setLineDash([]);

    // Utterance Polygon
    const scores = RADAR_AXES.map(a => Math.max(10, Math.min(100, radarData[a.key] || 20)) / 100);
    const isSarcastic = currentAnalysisData && currentAnalysisData.prediction.label === "SARCASTIC";

    radarCtx.beginPath();
    for (let i = 0; i < numAxes; i++) {
        const val = scores[i];
        const angle = (Math.PI * 2 / numAxes) * i - Math.PI / 2;
        const x = cx + Math.cos(angle) * (radius * val);
        const y = cy + Math.sin(angle) * (radius * val);
        if (i === 0) radarCtx.moveTo(x, y);
        else radarCtx.lineTo(x, y);
    }
    radarCtx.closePath();

    radarCtx.fillStyle = isSarcastic ? "rgba(244, 63, 94, 0.16)" : "rgba(16, 185, 129, 0.16)";
    radarCtx.fill();

    radarCtx.strokeStyle = isSarcastic ? "#f43f5e" : "#10b981";
    radarCtx.lineWidth = 2.2;
    radarCtx.stroke();

    // Data points
    for (let i = 0; i < numAxes; i++) {
        const val = scores[i];
        const angle = (Math.PI * 2 / numAxes) * i - Math.PI / 2;
        const x = cx + Math.cos(angle) * (radius * val);
        const y = cy + Math.sin(angle) * (radius * val);

        radarCtx.fillStyle = isSarcastic ? "#f43f5e" : "#10b981";
        radarCtx.beginPath();
        radarCtx.arc(x, y, 3.5, 0, Math.PI * 2);
        radarCtx.fill();
    }
}

// ===== Pitch Track (F0 Contour) =====
function drawIdlePitchGrid() {
    const w = pitchContourCanvas.getBoundingClientRect().width;
    const h = pitchContourCanvas.getBoundingClientRect().height;
    if (w === 0 || h === 0) return;

    pitchCanvasCtx.clearRect(0, 0, w, h);
    pitchCanvasCtx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    pitchCanvasCtx.lineWidth = 1;

    for (let y = 0; y <= h; y += 32) {
        pitchCanvasCtx.beginPath();
        pitchCanvasCtx.moveTo(0, y);
        pitchCanvasCtx.lineTo(w, y);
        pitchCanvasCtx.stroke();
    }
}

function drawPitchContour(contour, isPitchFlat) {
    if (!contour || contour.length === 0) return;

    const w = pitchContourCanvas.getBoundingClientRect().width;
    const h = pitchContourCanvas.getBoundingClientRect().height;
    if (w === 0 || h === 0) return;

    pitchCanvasCtx.clearRect(0, 0, w, h);

    // Frequency Level Grid
    pitchCanvasCtx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    pitchCanvasCtx.fillStyle = "#64748b";
    pitchCanvasCtx.font = "10px 'JetBrains Mono', monospace";

    const hzLevels = [100, 150, 200, 250, 300];
    const minHz = 75;
    const maxHz = 325;

    for (const hz of hzLevels) {
        const y = h - ((hz - minHz) / (maxHz - minHz)) * h;
        pitchCanvasCtx.beginPath();
        pitchCanvasCtx.moveTo(42, y);
        pitchCanvasCtx.lineTo(w, y);
        pitchCanvasCtx.stroke();
        pitchCanvasCtx.fillText(`${hz}Hz`, 6, y + 3);
    }

    const paddingLeft = 45;
    const plotWidth = w - paddingLeft - 15;
    const strokeColor = isPitchFlat ? "#f43f5e" : "#3b82f6";

    // Curve rendering
    pitchCanvasCtx.lineWidth = 2.4;
    pitchCanvasCtx.strokeStyle = strokeColor;
    pitchCanvasCtx.beginPath();

    let started = false;
    for (let i = 0; i < contour.length; i++) {
        const p = contour[i];
        if (p.voiced && p.hz > 50) {
            const x = paddingLeft + (i / (contour.length - 1)) * plotWidth;
            const clampedHz = Math.max(minHz, Math.min(maxHz, p.hz));
            const y = h - ((clampedHz - minHz) / (maxHz - minHz)) * h;

            if (!started) {
                pitchCanvasCtx.moveTo(x, y);
                started = true;
            } else {
                pitchCanvasCtx.lineTo(x, y);
            }
        }
    }
    pitchCanvasCtx.stroke();

    // Data points along contour
    for (let i = 0; i < contour.length; i++) {
        const p = contour[i];
        if (p.voiced && p.hz > 50) {
            const x = paddingLeft + (i / (contour.length - 1)) * plotWidth;
            const clampedHz = Math.max(minHz, Math.min(maxHz, p.hz));
            const y = h - ((clampedHz - minHz) / (maxHz - minHz)) * h;

            pitchCanvasCtx.fillStyle = strokeColor;
            pitchCanvasCtx.beginPath();
            pitchCanvasCtx.arc(x, y, 2.5, 0, Math.PI * 2);
            pitchCanvasCtx.fill();
        }
    }
}

// ===== WAV PCM Resampler & Encoder =====
function encodeWAV(buffers, sampleRate) {
    const totalLength = buffers.reduce((acc, b) => acc + b.length, 0);
    const samples = new Float32Array(totalLength);
    let offset = 0;
    for (const b of buffers) {
        samples.set(b, offset);
        offset += b.length;
    }

    const targetRate = 16000;
    const resampled = (sampleRate === targetRate) ? samples : resampleAudio(samples, sampleRate, targetRate);

    const buffer = new ArrayBuffer(44 + resampled.length * 2);
    const view = new DataView(buffer);

    function writeString(view, offset, str) {
        for (let i = 0; i < str.length; i++) {
            view.setUint8(offset + i, str.charCodeAt(i));
        }
    }

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + resampled.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, targetRate, true);
    view.setUint32(28, targetRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, resampled.length * 2, true);

    let pcmOffset = 44;
    for (let i = 0; i < resampled.length; i++, pcmOffset += 2) {
        const s = Math.max(-1, Math.min(1, resampled[i]));
        view.setInt16(pcmOffset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

function resampleAudio(audioBuffer, fromSampleRate, toSampleRate) {
    if (fromSampleRate === toSampleRate) return audioBuffer;
    const compression = fromSampleRate / toSampleRate;
    const length = Math.round(audioBuffer.length / compression);
    const result = new Float32Array(length);
    for (let i = 0; i < length; i++) {
        const index = Math.round(i * compression);
        result[i] = audioBuffer[index] || 0;
    }
    return result;
}

// ===== Recording Lifecycle =====
async function startRecording() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
        });

        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(mediaStream);

        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);

        scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
        pcmBuffers = [];
        scriptProcessor.onaudioprocess = (e) => {
            if (!isRecording) return;
            const input = e.inputBuffer.getChannelData(0);
            pcmBuffers.push(new Float32Array(input));
        };

        source.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);

        isRecording = true;
        hasSpoken = false;
        silenceStartTime = null;

        recordBtn.classList.add("recording");
        recordBtn.querySelector(".btn-text").textContent = "Stop Ingestion";

        systemStatus.querySelector(".status-text").textContent = "Recording...";
        systemStatus.querySelector(".status-pulse-dot").style.backgroundColor = "#f43f5e";

        recordingBadge.classList.remove("hidden");
        visualizerOverlay.style.opacity = "0";

        recordingStartTime = Date.now();
        startTimer();

        drawLiveWaveform();

    } catch (err) {
        console.error("Audio device access error:", err);
        showToast("Microphone access unavailable. Select a preset benchmark sample.");
    }
}

function stopRecording() {
    if (!isRecording) return;
    isRecording = false;
    stopTimer();

    recordBtn.classList.remove("recording");
    recordBtn.querySelector(".btn-text").textContent = "Record Voice";

    systemStatus.querySelector(".status-text").textContent = "Processing...";
    systemStatus.querySelector(".status-pulse-dot").style.backgroundColor = "#3b82f6";

    recordingBadge.classList.add("hidden");
    visualizerOverlay.style.opacity = "1";
    vadLevel.style.width = "0%";
    vadStatus.textContent = "Ingesting signal into inference pipeline...";

    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
    drawIdleWaveform();

    const sampleRate = audioContext ? audioContext.sampleRate : 16000;
    const wavBlob = encodeWAV(pcmBuffers, sampleRate);

    setupAudioPlayback(URL.createObjectURL(wavBlob));

    if (mediaStream) {
        mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
    }
    if (scriptProcessor) {
        scriptProcessor.disconnect();
        scriptProcessor = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }

    sendAudioForInference(wavBlob);
}

function startTimer() {
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const mins = Math.floor(elapsed / 60);
        const secs = elapsed % 60;
        recordTimer.textContent = `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
    }, 1000);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
    recordTimer.textContent = "00:00";
}

// ===== API Inference Client =====
async function sendAudioForInference(audioBlob) {
    loadingSection.classList.remove("hidden");
    resultsSection.classList.add("hidden");
    recordBtn.disabled = true;

    try {
        const formData = new FormData();
        formData.append("audio", audioBlob, "speech.wav");

        const resp = await fetch("/api/analyze", {
            method: "POST",
            body: formData,
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.error || "Inference error");
        }

        const data = await resp.json();
        renderAnalysis(data);

    } catch (err) {
        console.error("Pipeline inference failure:", err);
        showToast(`Inference Failure: ${err.message}`);
    } finally {
        loadingSection.classList.add("hidden");
        recordBtn.disabled = false;
        systemStatus.querySelector(".status-text").textContent = "Ready";
        systemStatus.querySelector(".status-pulse-dot").style.backgroundColor = "#10b981";
    }
}

// ===== Studio Audio Player Synchronization =====
function setupAudioPlayback(audioUrl) {
    if (!audioUrl) return;

    coreAudioElement.src = audioUrl;
    coreAudioElement.load();

    coreAudioElement.onloadedmetadata = () => {
        const dur = coreAudioElement.duration || 0;
        totalDurationEl.textContent = formatAudioTime(dur);
        audioScrubber.max = dur.toFixed(2);
        audioScrubber.value = 0;
    };

    coreAudioElement.ontimeupdate = () => {
        const cur = coreAudioElement.currentTime || 0;
        const dur = coreAudioElement.duration || 1;
        currentTimeEl.textContent = formatAudioTime(cur);
        audioScrubber.value = cur.toFixed(2);
        updateLaserCursor(cur / dur);
    };

    coreAudioElement.onplay = () => {
        playIcon.textContent = "❚❚";
        audioPlayToggle.classList.add("playing");
        heroPlayBtn.querySelector(".btn-text").textContent = "Pause Audio";
        pitchLaserCursor.style.opacity = "1";
    };

    coreAudioElement.onpause = () => {
        playIcon.textContent = "▶";
        audioPlayToggle.classList.remove("playing");
        heroPlayBtn.querySelector(".btn-text").textContent = "Play Audio";
    };

    coreAudioElement.onended = () => {
        playIcon.textContent = "▶";
        audioPlayToggle.classList.remove("playing");
        heroPlayBtn.querySelector(".btn-text").textContent = "Play Audio";
        pitchLaserCursor.style.opacity = "0";
        audioScrubber.value = 0;
        currentTimeEl.textContent = "00:00.0";
    };
}

function updateLaserCursor(fraction) {
    const pRect = pitchContourCanvas.getBoundingClientRect();
    if (pRect.width === 0) return;

    const paddingLeft = 45;
    const plotWidth = pRect.width - paddingLeft - 15;
    const targetX = paddingLeft + (fraction * plotWidth);

    pitchLaserCursor.style.left = `${targetX}px`;
}

function formatAudioTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 10);
    return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}.${ms}`;
}

function toggleAudioPlayback() {
    if (!coreAudioElement.src) return;
    if (coreAudioElement.paused) {
        coreAudioElement.play().catch(e => console.warn("Audio playback error:", e));
    } else {
        coreAudioElement.pause();
    }
}

audioPlayToggle.addEventListener("click", toggleAudioPlayback);
heroPlayBtn.addEventListener("click", toggleAudioPlayback);

audioScrubber.addEventListener("input", (e) => {
    if (coreAudioElement.duration) {
        coreAudioElement.currentTime = parseFloat(e.target.value);
    }
});

// Click-to-seek on Pitch Contour Canvas
pitchContourCanvas.addEventListener("click", (e) => {
    if (!coreAudioElement.duration) return;

    const pRect = pitchContourCanvas.getBoundingClientRect();
    const clickX = e.clientX - pRect.left;
    const paddingLeft = 45;
    const plotWidth = pRect.width - paddingLeft - 15;

    if (clickX >= paddingLeft && clickX <= paddingLeft + plotWidth) {
        const fraction = (clickX - paddingLeft) / plotWidth;
        coreAudioElement.currentTime = fraction * coreAudioElement.duration;
        updateLaserCursor(fraction);
    }
});

// Playback Speed Controls
document.querySelectorAll(".speed-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".speed-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const speed = parseFloat(btn.dataset.speed);
        coreAudioElement.playbackRate = speed;
        showToast(`Playback speed set to ${speed}×`);
    });
});

// ===== Results Render Engine =====
function renderAnalysis(data) {
    currentAnalysisData = data;
    resultsSection.classList.remove("hidden");

    transcriptInput.value = data.transcription;

    if (data.audio_url) {
        setupAudioPlayback(data.audio_url);
    }

    const isSarcastic = data.prediction.label === "SARCASTIC";
    const prob = data.prediction.sarcasm_probability;
    const conf = data.prediction.confidence;

    // 1. Primary Verdict Banner
    verdictHeroCard.className = `verdict-banner-card ${isSarcastic ? "verdict-sarcastic" : "verdict-genuine"}`;
    verdictHeadline.textContent = isSarcastic ? "SARCASTIC" : "GENUINE / NON-SARCASTIC";
    verdictBeacon.textContent = isSarcastic ? "Sarcastic Intent Detected" : "Genuine Intent Verified";

    // SVG Gauge Progress Ring (circumference = 2 * PI * 66 ≈ 414.69)
    const circumference = 414.69;
    const ringOffset = circumference * (1 - (prob / 100));
    ringProgress.style.strokeDashoffset = ringOffset;
    animateCounter(sarcasmProb, prob, "%");

    confidenceValue.textContent = `Confidence: ${conf}%`;
    confidenceValue.className = `enterprise-pill ${isSarcastic ? "badge-sarcastic" : "badge-genuine"}`;

    // Delivery Archetype
    if (data.archetype) {
        archetypeIcon.textContent = data.archetype.icon || "🧊";
        archetypeName.textContent = data.archetype.name || "Deadpan Delivery";
        archetypeDescription.textContent = data.archetype.description || "";
    }

    // Discrepancy / Conflict Status
    const confAnalysis = data.conflict_analysis;
    if (confAnalysis) {
        conflictChip.textContent = `${confAnalysis.conflict_level} Discrepancy`;
        if (confAnalysis.conflict_level === "CRITICAL") {
            conflictChip.className = "badge-conflict-marker badge-critical";
        } else if (confAnalysis.conflict_level === "MODERATE") {
            conflictChip.className = "badge-conflict-marker";
        } else {
            conflictChip.className = "badge-conflict-marker badge-harmonic";
        }
    }

    // 2. Lexical vs Acoustic Saliency (Token Attribution)
    renderTokenSaliency(data.saliency_analysis);

    // Emojis
    const emojiInfo = data.emoji_analysis;
    if (emojiInfo && emojiInfo.has_emojis && emojiInfo.detected.length > 0) {
        emojiRecognitionBar.classList.remove("hidden");
        emojiTags.innerHTML = "";
        for (const item of emojiInfo.detected) {
            const tag = document.createElement("span");
            tag.className = "enterprise-pill";
            tag.innerHTML = `<strong>${item.emoji}</strong> ${item.name} (${Math.round(item.sarcasm_weight * 100)}% bias)`;
            emojiTags.appendChild(tag);
        }
    } else {
        emojiRecognitionBar.classList.add("hidden");
    }

    // 3. Prosody Radar
    if (data.radar_data) {
        drawRadarChart(data.radar_data);
        valFlatness.textContent = `${data.radar_data.pitch_flatness}%`;
        valHyperbole.textContent = `${data.radar_data.hyperbolic_intensity}%`;
        valDrawl.textContent = `${data.radar_data.vocal_drawl}%`;
        valContrast.textContent = `${data.radar_data.acoustic_contrast}%`;
        valDivergence.textContent = `${data.radar_data.cross_modal_divergence}%`;
        valMarkers.textContent = `${data.radar_data.ironic_marker_load}%`;
    }

    // 4. Pitch Track
    currentPitchContour = data.pitch_contour || [];
    isPitchCurrentlyFlat = confAnalysis ? confAnalysis.is_pitch_flat : (data.audio_analysis.pitch.label === "Flat");
    drawPitchContour(currentPitchContour, isPitchCurrentlyFlat);

    // 5. MFCC Spectrum Analyzer
    renderMFCCBands(data.audio_analysis.mfcc_bands);

    // 6. Acoustic Telemetry Stats
    const a = data.audio_analysis;
    pitchBadge.textContent = a.pitch.label;
    pitchBadge.className = a.pitch.label === "Flat" ? "metric-pill badge-sarcastic" : "metric-pill";
    pitchValue.textContent = `${Math.round(a.pitch_mean_hz)} Hz`;
    pitchStdVal.textContent = `${a.pitch_std_hz.toFixed(1)} Hz`;
    pitchRangeVal.textContent = `${(a.pitch_range_hz || 0).toFixed(1)} Hz`;
    pitchDetail.textContent = a.pitch.detail;

    energyBadge.textContent = a.energy.label;
    energyValue.textContent = `${a.energy_rms.toFixed(3)} RMS`;
    energyPeakVal.textContent = `${(a.energy_rms_max || a.energy_rms * 1.5).toFixed(3)} RMS`;
    energyDetail.textContent = a.energy.detail;

    paceBadge.textContent = a.pace.label;
    tempoValue.textContent = `${Math.round(a.tempo_bpm)} BPM`;
    drawlVal.textContent = a.tempo_bpm < 95 ? "High (Drawl)" : "Standard";
    paceDetail.textContent = a.pace.detail;

    const centroid = a.spectral_centroid_mean || 1800;
    centroidValue.textContent = `${Math.round(centroid).toLocaleString()} Hz`;
    zcrVal.textContent = (a.zero_crossing_rate_mean || 0.08).toFixed(3);
    timbreBadge.textContent = centroid > 2200 ? "Bright" : (centroid < 1400 ? "Dark" : "Balanced");
    timbreDetail.textContent = `Centroid: ${Math.round(centroid)} Hz · Zero Crossing: ${(a.zero_crossing_rate_mean || 0.08).toFixed(3)}`;

    // 7. Modality Influence & Discrepancy
    const latent = data.latent_diagnostics;
    if (latent) {
        dominanceChip.textContent = `${latent.dominant_modality.split(' ')[0]} Dominant`;
        textInfluenceScore.textContent = `${latent.text_influence_pct}%`;
        textInfluenceBar.style.width = `${latent.text_influence_pct}%`;
        audioInfluenceScore.textContent = `${latent.audio_influence_pct}%`;
        audioInfluenceBar.style.width = `${latent.audio_influence_pct}%`;
    }

    if (confAnalysis) {
        conflictScore.textContent = `${confAnalysis.conflict_index}%`;
        conflictDetail.textContent = confAnalysis.detail;
        setTimeout(() => {
            conflictMeter.style.width = `${confAnalysis.conflict_index}%`;
        }, 150);
    }

    // 8. Ablation Ensemble
    const comp = data.model_comparison;
    renderAblationRow(textMeter, textScore, textChip, comp.text_only);
    renderAblationRow(audioMeter, audioScore, audioChip, comp.audio_only);
    renderAblationRow(fusionMeter, fusionScore, fusionChip, comp.fusion);

    // Smooth scroll to findings
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ===== Token Saliency Inspector =====
function renderTokenSaliency(saliency) {
    if (!saliency) return;

    tokenStream.innerHTML = "";
    const tokens = saliency.tokens || [];

    tokens.forEach((item) => {
        const chip = document.createElement("div");
        const isPos = item.lexical_valence > 0 || item.category === "praise";
        const isNeg = item.lexical_valence < 0;

        chip.className = `token-chip ${isPos ? "token-pos" : (isNeg ? "token-neg" : "token-neu")}`;

        let subtagHtml = "";
        if (item.category === "praise") {
            subtagHtml = `<span class="token-badge">+Praise</span>`;
        } else if (item.category === "intensifier") {
            subtagHtml = `<span class="token-badge">Amp</span>`;
        }

        chip.innerHTML = `<span>${item.token}</span> ${subtagHtml}`;

        chip.addEventListener("mouseenter", () => {
            document.querySelectorAll(".token-chip").forEach(c => c.classList.remove("token-active"));
            chip.classList.add("token-active");
            inspectToken(item);
        });

        tokenStream.appendChild(chip);
    });

    lexicalValence.textContent = `${saliency.lexical_valence >= 0 ? "+" : ""}${saliency.lexical_valence.toFixed(2)}`;
    if (saliency.lexical_valence > 0.4) {
        lexicalTag.textContent = "Positive Praise";
    } else if (saliency.lexical_valence < -0.2) {
        lexicalTag.textContent = "Negative Context";
    } else {
        lexicalTag.textContent = "Neutral Context";
    }

    acousticValence.textContent = `${saliency.acoustic_valence >= 0 ? "+" : ""}${saliency.acoustic_valence.toFixed(2)}`;
    acousticTag.textContent = saliency.acoustic_valence < -0.4 ? "Flat / Monotone" : "Dynamic Inflection";

    polarityGap.textContent = `${saliency.polarity_gap.toFixed(2)}Δ`;
    if (saliency.polarity_gap > 1.2) {
        polarityGapTag.textContent = "Critical Discrepancy";
    } else if (saliency.polarity_gap > 0.6) {
        polarityGapTag.textContent = "Moderate Divergence";
    } else {
        polarityGapTag.textContent = "Harmonic Delivery";
    }

    saliencyNarrative.textContent = saliency.narrative || "";
}

function inspectToken(item) {
    const isContradiction = item.acoustic_alignment === "CONTRADICTION";
    tokenInspectorBox.innerHTML = `
        <div style="display: flex; gap: 1.25rem; align-items: center; flex-wrap: wrap; width: 100%;">
            <div><span style="color: var(--text-tertiary); font-size: 0.72rem;">WORD:</span> <strong style="color: #ffffff;">"${item.token}"</strong></div>
            <div><span style="color: var(--text-tertiary); font-size: 0.72rem;">CATEGORY:</span> <strong style="color: var(--text-secondary);">${item.category}</strong></div>
            <div><span style="color: var(--text-tertiary); font-size: 0.72rem;">LEXICAL SCORE:</span> <strong style="color: ${item.lexical_valence >= 0 ? '#6ee7b7' : '#fca5a5'};">${item.lexical_valence >= 0 ? '+' : ''}${item.lexical_valence}</strong></div>
            <div><span style="color: var(--text-tertiary); font-size: 0.72rem;">SALIENCY:</span> <strong style="color: #ffffff;">${Math.round(item.sarcasm_saliency * 100)}%</strong></div>
            <div><span class="enterprise-pill ${isContradiction ? 'badge-sarcastic' : 'badge-genuine'}">${item.acoustic_alignment}</span></div>
        </div>
    `;
}

// ===== MFCC Spectrum Equalizer =====
function renderMFCCBands(bands) {
    if (!bands || bands.length === 0) return;

    mfccBarsContainer.innerHTML = "";
    bands.forEach(b => {
        const col = document.createElement("div");
        col.className = "mfcc-col";
        col.title = `${b.label}: ${b.val} (${b.power_pct}%)`;

        col.innerHTML = `
            <div class="mfcc-meter">
                <div class="mfcc-fill" style="height: ${b.power_pct}%;"></div>
            </div>
            <span class="mfcc-id">M${b.index}</span>
        `;
        mfccBarsContainer.appendChild(col);
    });
}

// ===== Ablation Study Progress Rows =====
function renderAblationRow(meterEl, scoreEl, chipEl, modelData) {
    const isSarcastic = modelData.label === "SARCASTIC";

    setTimeout(() => {
        meterEl.style.width = `${modelData.sarcasm_probability}%`;
    }, 150);

    scoreEl.textContent = `${modelData.sarcasm_probability}%`;
    chipEl.textContent = modelData.label === "SARCASTIC" ? "Sarcastic" : "Genuine";
    chipEl.className = `score-badge ${isSarcastic ? "badge-sarcastic" : "badge-genuine"}`;
}

// Counter animation
function animateCounter(element, targetVal, suffix = "") {
    let current = 0;
    const duration = 500;
    const stepTime = 20;
    const steps = duration / stepTime;
    const increment = targetVal / steps;

    const timer = setInterval(() => {
        current += increment;
        if (current >= targetVal) {
            current = targetVal;
            clearInterval(timer);
        }
        element.textContent = `${current.toFixed(1)}${suffix}`;
    }, stepTime);
}

// Global Toast Feedback
function showToast(msg) {
    toastMessage.textContent = msg;
    toastNotification.classList.remove("hidden");
    if (toastTimeout) clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
        toastNotification.classList.add("hidden");
    }, 3200);
}

// Copy Markdown Report
copyReportBtn.addEventListener("click", () => {
    if (!currentAnalysisData) return;
    const d = currentAnalysisData;
    const report = `
# SarcasTone™ Enterprise Analysis Report
--------------------------------------------------
Classification: ${d.prediction.label}
Sarcasm Probability: ${d.prediction.sarcasm_probability}%
Confidence: ${d.prediction.confidence}%
Archetype: ${d.archetype ? d.archetype.name : "N/A"}
Transcript: "${d.transcription}"

## Acoustic Telemetry
- Fundamental Pitch (F0 Mean): ${Math.round(d.audio_analysis.pitch_mean_hz)} Hz
- Pitch Inflection Variance (F0 Std): ${d.audio_analysis.pitch_std_hz.toFixed(1)} Hz (${d.audio_analysis.pitch.label})
- RMS Energy: ${d.audio_analysis.energy_rms.toFixed(3)}
- Speaking Cadence: ${Math.round(d.audio_analysis.tempo_bpm)} BPM

## Cross-Modal Discrepancy
- Lexical Sentiment: ${d.saliency_analysis ? d.saliency_analysis.lexical_valence : "N/A"}
- Acoustic Pitch Valence: ${d.saliency_analysis ? d.saliency_analysis.acoustic_valence : "N/A"}
- Polarity Contrast Delta: ${d.saliency_analysis ? d.saliency_analysis.polarity_gap : "N/A"}Δ
- Discrepancy Index: ${d.conflict_analysis.conflict_index}% (${d.conflict_analysis.conflict_level})

## Model Ensemble Ablation
- DistilBERT 768-d Text Baseline: ${d.model_comparison.text_only.sarcasm_probability}%
- Librosa 26-d Audio Baseline: ${d.model_comparison.audio_only.sarcasm_probability}%
- SarcasTone 794-d Multimodal Fusion: ${d.model_comparison.fusion.sarcasm_probability}%
--------------------------------------------------
    `.trim();

    navigator.clipboard.writeText(report).then(() => {
        showToast("Analysis report copied to clipboard.");
    }).catch(() => {
        showToast("Clipboard copy failed.");
    });
});

// Export JSON
exportJsonBtn.addEventListener("click", () => {
    if (!currentAnalysisData) return;
    const jsonStr = JSON.stringify(currentAnalysisData, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `sarcastone_telemetry_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast("Downloaded telemetry JSON.");
});

// Counterfactual Re-evaluation
reevaluateBtn.addEventListener("click", async () => {
    const text = transcriptInput.value.trim();
    if (!text) return;

    loadingSection.classList.remove("hidden");
    resultsSection.classList.add("hidden");
    systemStatus.querySelector(".status-text").textContent = "Re-evaluating...";

    try {
        const resp = await fetch("/api/reevaluate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.error || "Evaluation error");
        }

        const data = await resp.json();
        renderAnalysis(data);
        showToast("Counterfactual utterance re-evaluated.");

    } catch (err) {
        console.error("Re-evaluation fault:", err);
        showToast(`Evaluation Error: ${err.message}`);
    } finally {
        loadingSection.classList.add("hidden");
        systemStatus.querySelector(".status-text").textContent = "Ready";
    }
});

// Emoji Modifiers
document.querySelectorAll(".emoji-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        transcriptInput.value += ` ${btn.dataset.emoji}`;
        transcriptInput.focus();
    });
});

// Preset Benchmark Samples
async function loadBenchmarkSample(id) {
    loadingSection.classList.remove("hidden");
    resultsSection.classList.add("hidden");
    systemStatus.querySelector(".status-text").textContent = `Loading #${id}...`;

    try {
        const resp = await fetch(`/api/sample/${id}`);
        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.error || "Failed to load sample");
        }
        const data = await resp.json();
        renderAnalysis(data);
        showToast(`Loaded Benchmark Sample #${id}`);
    } catch (err) {
        console.error("Benchmark sample error:", err);
        showToast(`Sample error: ${err.message}`);
    } finally {
        loadingSection.classList.add("hidden");
        systemStatus.querySelector(".status-text").textContent = "Ready";
    }
}

document.querySelectorAll(".preset-pill").forEach(btn => {
    btn.addEventListener("click", () => {
        loadBenchmarkSample(btn.dataset.id);
    });
});

// Local File Upload
fileUpload.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
        setupAudioPlayback(URL.createObjectURL(file));
        sendAudioForInference(file);
    }
});

// Reset Workbench Session
resetBtn.addEventListener("click", () => {
    resultsSection.classList.add("hidden");
    textMeter.style.width = "0%";
    audioMeter.style.width = "0%";
    fusionMeter.style.width = "0%";
    conflictMeter.style.width = "0%";
    currentAnalysisData = null;
    currentPitchContour = [];
    if (coreAudioElement) {
        coreAudioElement.pause();
        coreAudioElement.src = "";
    }
    drawIdleWaveform();
    drawIdlePitchGrid();
    drawIdleRadar();
    window.scrollTo({ top: 0, behavior: "smooth" });
    showToast("Workbench session reset.");
});

// Record Button
recordBtn.addEventListener("click", () => {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
});

// Lifecycle setup
window.addEventListener("load", setupCanvases);
window.addEventListener("resize", setupCanvases);
