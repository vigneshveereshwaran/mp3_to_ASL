/**
 * HearLink ASL — Client WebSocket Manager
 * Manages WebSocket connection, binary audio streaming, and incoming pose frame JSON messages.
 */

export class WebSocketClient {
    constructor(callbacks = {}) {
        this.callbacks = callbacks;
        this.socket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host || 'localhost:8000';
        const url = `${protocol}//${host}/ws/stream`;

        console.log(`[WebSocketClient] Connecting to ${url}...`);
        this.socket = new WebSocket(url);
        this.socket.binaryType = 'arraybuffer';

        this.socket.onopen = () => {
            console.log("[WebSocketClient] Connected to streaming server");
            this.isConnected = true;
            this.reconnectAttempts = 0;
            if (this.callbacks.onConnect) this.callbacks.onConnect();
        };

        this.socket.onmessage = (event) => {
            if (typeof event.data === 'string') {
                try {
                    const data = JSON.parse(event.data);
                    this._handleMessage(data);
                } catch (e) {
                    console.error("[WebSocketClient] JSON parse error:", e);
                }
            }
        };

        this.socket.onerror = (error) => {
            console.error("[WebSocketClient] Socket error:", error);
            if (this.callbacks.onError) this.callbacks.onError(error);
        };

        this.socket.onclose = () => {
            console.warn("[WebSocketClient] Connection closed");
            this.isConnected = false;
            if (this.callbacks.onDisconnect) this.callbacks.onDisconnect();
            this._tryReconnect();
        };
    }

    _tryReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`[WebSocketClient] Reconnecting attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}...`);
            setTimeout(() => this.connect(), 2000 * this.reconnectAttempts);
        }
    }

    sendAudioChunk(arrayBuffer) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(arrayBuffer);
        }
    }

    sendTextInput(text) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify({ type: 'text_input', text }));
        }
    }

    sendReset() {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify({ type: 'reset' }));
        }
    }

    _handleMessage(data) {
        switch (data.type) {
            case 'transcript':
                if (this.callbacks.onTranscript) this.callbacks.onTranscript(data);
                break;
            case 'gloss':
                if (this.callbacks.onGloss) this.callbacks.onGloss(data);
                break;
            case 'pose_frames':
                if (this.callbacks.onPoseFrames) this.callbacks.onPoseFrames(data);
                break;
            case 'status':
                console.log("[WebSocket Server Status]:", data.message);
                break;
            default:
                console.log("[WebSocket Message]:", data);
        }
    }

    disconnect() {
        if (this.socket) {
            this.socket.close();
            this.socket = null;
        }
    }
}
