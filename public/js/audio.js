/**
 * HearLink ASL — WebRTC Audio Recorder
 * Captures microphone audio using getUserMedia and converts PCM Float32 to Int16 PCM.
 */

export class AudioRecorder {
    constructor(onAudioChunk) {
        this.onAudioChunk = onAudioChunk;
        this.mediaStream = null;
        this.audioContext = null;
        this.scriptProcessor = null;
        this.isRecording = false;
        this.sampleRate = 16000;
    }

    async start() {
        if (this.isRecording) return;

        try {
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: this.sampleRate,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });

            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: this.sampleRate
            });

            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            this.scriptProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);

            this.scriptProcessor.onaudioprocess = (event) => {
                if (!this.isRecording) return;
                const inputData = event.inputBuffer.getChannelData(0);
                const pcm16Data = this._convertFloat32ToInt16(inputData);
                if (this.onAudioChunk) {
                    this.onAudioChunk(pcm16Data.buffer);
                }
            };

            source.connect(this.scriptProcessor);
            this.scriptProcessor.connect(this.audioContext.destination);
            this.isRecording = true;
            console.log("[AudioRecorder] Microphoned started recording at 16kHz");
            return true;
        } catch (err) {
            console.error("[AudioRecorder] Microphone access failed:", err);
            throw err;
        }
    }

    stop() {
        if (!this.isRecording) return;

        this.isRecording = false;

        if (this.scriptProcessor) {
            this.scriptProcessor.disconnect();
            this.scriptProcessor = null;
        }

        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        console.log("[AudioRecorder] Microphone stopped");
    }

    _convertFloat32ToInt16(buffer) {
        let l = buffer.length;
        let buf = new Int16Array(l);
        while (l--) {
            let s = Math.max(-1, Math.min(1, buffer[l]));
            buf[l] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return buf;
    }
}
