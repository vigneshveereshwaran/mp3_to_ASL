/**
 * MP4-to-ASL — Main Application Controller
 * Synchronizes MP4 video timestamp playback, speech transcription, Gloss tokens,
 * and 3D Avatar skeletal sign animations.
 */

class MP4ToASLApp {
    constructor() {
        this.avatarRenderer = null;
        this.timeline = [];
        this.activeSegmentIndex = -1;
        this.isVideoLoaded = false;
        this.isRecording = false;
        this.mediaRecorder = null;
        this._init();
    }

    async _init() {
        // Initialize 3D Avatar Renderer
        const canvas = document.getElementById('three-canvas');
        if (canvas && window.AvatarRenderer) {
            this.avatarRenderer = new window.AvatarRenderer(canvas);
        }

        // Setup Event Handlers
        this._setupVideoEvents();
        this._setupInputEvents();
        this._checkHealth();

        // Hide loading overlay
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            setTimeout(() => overlay.classList.add('hidden'), 500);
        }
    }

    async _checkHealth() {
        try {
            const res = await fetch('/health');
            const data = await res.json();
            const dot = document.getElementById('backend-status-dot');
            const txt = document.getElementById('backend-type-label');
            if (dot) dot.classList.add('online');
            if (txt) txt.textContent = data.neural_translator ? 'Neural ASL' : 'Rule ASL';
        } catch {
            const txt = document.getElementById('backend-type-label');
            if (txt) txt.textContent = 'Offline';
        }
    }

    _setupVideoEvents() {
        const video = document.getElementById('mp4-video');
        const scrubber = document.getElementById('video-scrubber');
        const playBtn = document.getElementById('btn-play-pause');
        const fileInput = document.getElementById('video-file-input');

        if (video) {
            video.addEventListener('timeupdate', () => {
                const curTime = video.currentTime;
                const duration = video.duration || 1;

                if (scrubber) {
                    scrubber.value = (curTime / duration) * 100;
                }

                const timeDisp = document.getElementById('time-display');
                if (timeDisp) {
                    timeDisp.textContent = `${this._formatTime(curTime)} / ${this._formatTime(duration)}`;
                }

                this._syncTimelineWithTime(curTime);
            });

            video.addEventListener('ended', () => {
                if (playBtn) playBtn.textContent = '▶ Play';
                this._setSigningBadge(null);
            });
        }

        if (playBtn && video) {
            playBtn.addEventListener('click', () => {
                if (video.paused) {
                    video.play();
                    playBtn.textContent = '⏸ Pause';
                } else {
                    video.pause();
                    playBtn.textContent = '▶ Play';
                }
            });
        }

        if (scrubber && video) {
            scrubber.addEventListener('input', () => {
                const duration = video.duration || 1;
                video.currentTime = (scrubber.value / 100) * duration;
            });
        }

        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) this.uploadMP4Video(file);
            });
        }
    }

    _setupInputEvents() {
        const sendBtn = document.getElementById('btn-send-text');
        const textInput = document.getElementById('text-input');
        const presetSelect = document.getElementById('preset-select');
        const micBtn = document.getElementById('btn-mic-record');

        if (sendBtn && textInput) {
            sendBtn.addEventListener('click', () => this.translateCustomText());
            textInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') this.translateCustomText();
            });
        }

        if (presetSelect && textInput) {
            presetSelect.addEventListener('change', (e) => {
                const val = e.target.value;
                if (val) {
                    textInput.value = val;
                    this.translateCustomText();
                    e.target.value = '';
                }
            });
        }

        if (micBtn) {
            micBtn.addEventListener('click', () => this.toggleMicRecord());
        }
    }

    async uploadMP4Video(file) {
        const formData = new FormData();
        formData.append('file', file);

        const videoPlayer = document.getElementById('mp4-video');
        const placeholder = document.getElementById('video-placeholder');

        if (placeholder) placeholder.style.display = 'none';
        if (videoPlayer) {
            videoPlayer.style.display = 'block';
            videoPlayer.src = URL.createObjectURL(file);
        }

        this._updateCaption("Processing MP4 video & audio stream...", []);

        try {
            const res = await fetch('/api/video/process', {
                method: 'POST',
                body: formData
            });

            const data = await res.json();
            if (data.status === 'success') {
                this.timeline = data.timeline || [];
                this.isVideoLoaded = true;
                this.activeSegmentIndex = -1;

                this._updateCaption(
                    data.full_english_text || "MP4 video ready for playback",
                    ["READY", "SYNCED"]
                );

                if (videoPlayer) {
                    videoPlayer.play();
                    const playBtn = document.getElementById('btn-play-pause');
                    if (playBtn) playBtn.textContent = '⏸ Pause';
                }
            } else {
                alert("Video processing failed: " + (data.detail || "Unknown error"));
            }
        } catch (err) {
            console.error("Upload error:", err);
            this._updateCaption("Error processing video file.", ["ERROR"]);
        }
    }

    async translateCustomText() {
        const textInput = document.getElementById('text-input');
        const text = textInput ? textInput.value.trim() : '';
        if (!text) return;

        this._updateCaption(text, ["TRANSLATING..."]);

        try {
            const res = await fetch('/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            const data = await res.json();
            const tokens = data.tokens || (data.gloss ? data.gloss.split(' ') : []);
            this._updateCaption(data.english, tokens);

            // Play multi-sign 3D animation sequence
            if (this.avatarRenderer && tokens.length > 0) {
                this.avatarRenderer.playSignSequence(tokens, (activeSign) => {
                    this._setSigningBadge(activeSign);
                    this._highlightActiveGlossChip(activeSign);
                });
            }
        } catch (err) {
            console.error("Translation error:", err);
            this._updateCaption(text, ["SERVER-ERROR"]);
        }
    }

    _syncTimelineWithTime(currentTime) {
        if (!this.timeline || this.timeline.length === 0) return;

        for (let i = 0; i < this.timeline.length; i++) {
            const seg = this.timeline[i];
            if (currentTime >= seg.start && currentTime <= seg.end) {
                if (this.activeSegmentIndex !== i) {
                    this.activeSegmentIndex = i;
                    const tokens = seg.tokens || (seg.gloss ? seg.gloss.split(' ') : []);
                    this._updateCaption(seg.text, tokens, i);

                    if (this.avatarRenderer && tokens.length > 0) {
                        this.avatarRenderer.playSignSequence(tokens, (activeSign) => {
                            this._setSigningBadge(activeSign);
                            this._highlightActiveGlossChip(activeSign);
                        });
                    }
                }
                return;
            }
        }
    }

    _setSigningBadge(signName) {
        const badge = document.getElementById('signing-indicator');
        if (!badge) return;

        if (signName) {
            badge.textContent = `✋ Signing: ${signName}`;
            badge.classList.add('active');
        } else {
            badge.textContent = `✋ Signing ASL...`;
            badge.classList.remove('active');
        }
    }

    _highlightActiveGlossChip(activeSign) {
        const container = document.getElementById('caption-gloss-container');
        if (!container) return;

        const chips = container.querySelectorAll('.gloss-chip');
        chips.forEach(chip => {
            if (activeSign && chip.textContent.toUpperCase() === activeSign.toUpperCase()) {
                chip.classList.add('active');
            } else {
                chip.classList.remove('active');
            }
        });
    }

    _updateCaption(englishText, glossTokens) {
        const engBox = document.getElementById('caption-english-text');
        const glossBox = document.getElementById('caption-gloss-container');

        if (engBox) engBox.textContent = englishText || 'Ready for input...';

        if (glossBox) {
            glossBox.innerHTML = '';
            (glossTokens || []).forEach(token => {
                const chip = document.createElement('span');
                chip.className = 'gloss-chip';
                chip.textContent = token;
                glossBox.appendChild(chip);
            });
        }
    }

    _formatTime(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s < 10 ? '0' : ''}${s}`;
    }

    async toggleMicRecord() {
        const micBtn = document.getElementById('btn-mic-record');
        if (!this.isRecording) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                this.mediaRecorder = new MediaRecorder(stream);
                this.mediaRecorder.start();
                this.isRecording = true;
                if (micBtn) micBtn.className = 'btn btn-danger';

                if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
                    this.recognition = new SR();
                    this.recognition.lang = 'en-US';
                    this.recognition.onresult = (e) => {
                        const transcript = Array.from(e.results).map(r => r[0].transcript).join('');
                        const textInput = document.getElementById('text-input');
                        if (textInput) textInput.value = transcript;
                        if (e.results[e.results.length - 1].isFinal) {
                            this.translateCustomText();
                        }
                    };
                    this.recognition.start();
                }
            } catch (err) {
                alert("Microphone access denied or unavailable.");
            }
        } else {
            this.isRecording = false;
            if (this.mediaRecorder) this.mediaRecorder.stop();
            if (this.recognition) this.recognition.stop();
            if (micBtn) micBtn.className = 'btn btn-primary';
        }
    }
}

window.addEventListener('DOMContentLoaded', () => {
    window.app = new MP4ToASLApp();
});
