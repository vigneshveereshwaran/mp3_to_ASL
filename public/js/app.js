/**
 * HearLink ASL — Main Application Controller
 * Orchestrates Audio Recording, WebSocket communication, WASM Animation processing,
 * Preset Selectors, and 3D Avatar Rendering.
 */

import { AudioRecorder } from './audio.js';
import { WebSocketClient } from './websocket.js';
import { WASMAnimationBridge } from './wasm_bridge.js';
import { AvatarRenderer } from './avatar.js';

class HearLinkApp {
    constructor() {
        this.audioRecorder = null;
        this.wsClient = null;
        this.wasmBridge = null;
        this.avatarRenderer = null;

        this.frameQueue = [];
        this.isPlayingAnimation = false;

        // UI elements
        this.btnRecord = document.getElementById('btn-record');
        this.btnSend = document.getElementById('btn-send');
        this.txtInput = document.getElementById('text-input');
        this.presetSelect = document.getElementById('preset-select');
        this.transcriptBox = document.getElementById('transcript-box');
        this.glossBox = document.getElementById('gloss-box');
        this.statusIndicator = document.getElementById('status-indicator');
        this.statusText = document.getElementById('status-text');
        this.metricAsr = document.getElementById('metric-asr');
        this.metricGloss = document.getElementById('metric-gloss');
        this.metricFps = document.getElementById('metric-fps');

        this.init();
    }

    async init() {
        console.log("[HearLinkApp] Initializing HearLink ASL Web Client...");

        // 1. WASM Engine Initialization
        this.wasmBridge = new WASMAnimationBridge();
        await this.wasmBridge.init();

        // 2. 3D Avatar Scene Setup
        const canvasContainer = document.getElementById('viewport-container');
        this.avatarRenderer = new AvatarRenderer(canvasContainer);

        // 3. WebSocket Client Setup
        this.wsClient = new WebSocketClient({
            onConnect: () => this.updateStatus(true, "Connected"),
            onDisconnect: () => this.updateStatus(false, "Disconnected"),
            onTranscript: (data) => this.handleTranscript(data),
            onGloss: (data) => this.handleGloss(data),
            onPoseFrames: (data) => this.handlePoseFrames(data)
        });
        this.wsClient.connect();

        // 4. Audio Recorder Setup
        this.audioRecorder = new AudioRecorder((pcmBuffer) => {
            this.wsClient.sendAudioChunk(pcmBuffer);
        });

        // 5. Attach Event Listeners
        this.btnRecord.addEventListener('click', () => this.toggleRecord());
        this.btnSend.addEventListener('click', () => this.sendText());
        this.txtInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendText();
        });

        if (this.presetSelect) {
            this.presetSelect.addEventListener('change', (e) => {
                const text = e.target.value;
                if (text) {
                    this.txtInput.value = text;
                    this.sendText();
                }
            });
        }

        // 6. Start Pose Queue Consumer Loop (60 FPS)
        this.startAnimationLoop();
    }

    updateStatus(online, text) {
        if (online) {
            this.statusIndicator.classList.add('online');
        } else {
            this.statusIndicator.classList.remove('online');
        }
        this.statusText.textContent = text;
    }

    async toggleRecord() {
        if (!this.audioRecorder.isRecording) {
            try {
                await this.audioRecorder.start();
                this.btnRecord.classList.remove('btn-primary');
                this.btnRecord.classList.add('btn-danger');
                this.btnRecord.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
                    Stop Listening
                `;
            } catch (e) {
                alert("Microphone permission denied or unsupported.");
            }
        } else {
            this.audioRecorder.stop();
            this.btnRecord.classList.remove('btn-danger');
            this.btnRecord.classList.add('btn-primary');
            this.btnRecord.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/></svg>
                Speak English
            `;
        }
    }

    sendText() {
        const raw = this.txtInput.value;
        const text = raw.trim ? raw.trim() : raw;
        if (text) {
            this.frameQueue = []; // Clear previous animation queue
            this.wsClient.sendTextInput(text);
        }
    }

    handleTranscript(data) {
        this.transcriptBox.textContent = data.text || "...";
        if (data.latency_ms) {
            this.metricAsr.textContent = `${Math.round(data.latency_ms)}ms`;
        }
    }

    handleGloss(data) {
        this.glossBox.innerHTML = '';
        const tokens = data.tokens || [];
        tokens.forEach(token => {
            const chip = document.createElement('span');
            chip.className = 'gloss-chip';
            chip.textContent = token;
            this.glossBox.appendChild(chip);
        });

        if (data.latency_ms) {
            this.metricGloss.textContent = `${Math.round(data.latency_ms)}ms`;
        }
    }

    handlePoseFrames(data) {
        if (data.frames && data.frames.length > 0) {
            // Push incoming pose keypoint frames into playback queue
            this.frameQueue.push(...data.frames);
        }
    }

    startAnimationLoop() {
        let lastFrameTime = performance.now();
        let frameCount = 0;
        let fpsTimer = performance.now();
        let frameDelay = 0;

        const renderStep = () => {
            const now = performance.now();
            const dt = (now - lastFrameTime) / 1000;
            lastFrameTime = now;

            // Compute FPS
            frameCount++;
            if (now - fpsTimer >= 1000) {
                this.metricFps.textContent = `${frameCount}`;
                frameCount = 0;
                fpsTimer = now;
            }

            // Play pose frames with controlled pacing (~15-20 FPS sign frame rate)
            frameDelay += dt;
            if (this.frameQueue.length > 0 && frameDelay >= 0.05) {
                frameDelay = 0;
                const rawFrame = this.frameQueue.shift();

                // Pass frame through WASM Animation Engine for SLERP/IK smoothing
                this.wasmBridge.processFrame(rawFrame, dt);

                // Render smoothed pose frame onto Three.js avatar
                this.avatarRenderer.applyPoseFrame(rawFrame);
            }

            requestAnimationFrame(renderStep);
        };

        requestAnimationFrame(renderStep);
    }
}

// Instantiate App when DOM is ready
window.addEventListener('DOMContentLoaded', () => {
    window.app = new HearLinkApp();
});
